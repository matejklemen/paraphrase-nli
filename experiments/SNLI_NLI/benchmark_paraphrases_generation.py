import json
import logging
import os
import sys
from argparse import ArgumentParser
from time import time
import re

import numpy as np
import pandas as pd
import torch
from datasets import tqdm
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Subset, DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from src.data.cleaning import mask_not_na, inds_unique, mask_long_enough
from src.data.nli import TransformersSeqPairDataset
from src.models.nli_trainer import TransformersPITrainer

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--paraphrase_path", type=str,
					default="/home/matej/Documents/paraphrase-nli/experiments/SNLI_NLI/PARAPHRASE_IDENTIFICATION/snli-bin-roberta-base-argmax/all_para_id.csv")
parser.add_argument("--pretrained_name_or_path", type=str, default="gpt2")
parser.add_argument("--model_type", type=str, default="gpt2",
					choices=["gpt2"])

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_seq_len", type=int, default=52)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=3_000)
parser.add_argument("--random_seed", type=int, default=17)

parser.add_argument("--use_cpu", action="store_true")

# TODO: move training code to a TransformersPGTrainer
if __name__ == "__main__":
	args = parser.parse_args()
	DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
	if not os.path.exists(args.experiment_dir):
		os.makedirs(args.experiment_dir)

	if args.random_seed is not None:
		np.random.seed(args.random_seed)
		torch.manual_seed(args.random_seed)

	with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
		json.dump(vars(args), fp=f, indent=4)

	# Set up logging to file and stdout
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	for curr_handler in [logging.StreamHandler(sys.stdout),
						 logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
		curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
		logger.addHandler(curr_handler)

	for k, v in vars(args).items():
		v_str = str(v)
		v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
		logging.info(f"|{k:30s}|{v_str:50s}|")

	# No AutoTokenizerFast at the moment?
	if args.model_type == "gpt2":
		tokenizer_cls = GPT2TokenizerFast
	else:
		raise NotImplementedError(f"Model_type '{args.model_type}' is not supported")

	tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)
	tokenizer.add_special_tokens({
		"eos_token": "<EOS>",
		"pad_token": "<PAD>",
		"additional_special_tokens": ["<PARA>"]
	})
	SEPARATOR_ID = int(tokenizer.encode("<PARA>", add_special_tokens=False)[0])
	tokenizer.save_pretrained(args.experiment_dir)
	model = GPT2LMHeadModel.from_pretrained(args.pretrained_name_or_path).to(DEVICE)
	model.resize_token_embeddings(len(tokenizer))
	optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

	df = pd.read_csv(args.paraphrase_path)
	# Basic data cleaning - remove NAs (?), duplicate pairs, pairs with one sequence very short
	df = df.loc[mask_not_na(df["sequence1"], df["sequence2"])]
	df = df.iloc[inds_unique(df["sequence1"], df["sequence2"])]
	df = df.loc[mask_long_enough(df["sequence1"], df["sequence2"])]

	df = df.loc[df["label"] == 1].reset_index(drop=True)
	df["formatted"] = list(map(
		lambda pair: f"{pair[0]} <PARA> {pair[1]} {tokenizer.eos_token}",
		zip(df["sequence1"].tolist(), df["sequence2"].tolist())
	))
	num_ex = df.shape[0]

	indices = np.random.permutation(num_ex)
	train_df = df.iloc[indices[:int(0.7 * num_ex)]]
	dev_df = df.iloc[indices[int(0.7 * num_ex): int(0.85 * num_ex)]]
	test_df = df.iloc[indices[int(0.85 * num_ex):]]

	train_df.drop("formatted", axis=1).to_csv(os.path.join(args.experiment_dir, "train.csv"), sep=",", index=False)
	dev_df.drop("formatted", axis=1).to_csv(os.path.join(args.experiment_dir, "dev.csv"), sep=",", index=False)
	test_df.drop("formatted", axis=1).to_csv(os.path.join(args.experiment_dir, "test.csv"), sep=",", index=False)

	_encoded_train = tokenizer.batch_encode_plus(
		train_df["formatted"].tolist(),
		max_length=args.max_seq_len, padding="max_length", truncation="longest_first", return_tensors="pt"
	)
	_train_labels = _encoded_train["input_ids"].clone()
	for idx_ex in range(_train_labels.shape[0]):
		for idx_token in range(args.max_seq_len):
			_train_labels[idx_ex, idx_token] = -100
			if _encoded_train["input_ids"][idx_ex, idx_token] == SEPARATOR_ID:
				break
	_encoded_train["labels"] = _train_labels

	_encoded_dev = tokenizer.batch_encode_plus(
		dev_df["formatted"].tolist(),
		max_length=args.max_seq_len, padding="max_length", truncation="longest_first", return_tensors="pt"
	)
	_dev_labels = _encoded_dev["input_ids"].clone()
	for idx_ex in range(_dev_labels.shape[0]):
		for idx_token in range(args.max_seq_len):
			_dev_labels[idx_ex, idx_token] = -100
			if _encoded_dev["input_ids"][idx_ex, idx_token] == SEPARATOR_ID:
				break
	_encoded_dev["labels"] = _dev_labels

	_encoded_test = tokenizer.batch_encode_plus(
		test_df["formatted"].tolist(),
		max_length=args.max_seq_len, padding="max_length", truncation="longest_first", return_tensors="pt"
	)
	_test_labels = _encoded_test["input_ids"].clone()
	for idx_ex in range(_test_labels.shape[0]):
		for idx_token in range(args.max_seq_len):
			_test_labels[idx_ex, idx_token] = -100
			if _encoded_test["input_ids"][idx_ex, idx_token] == SEPARATOR_ID:
				break
	_encoded_test["labels"] = _test_labels

	train_set = TransformersSeqPairDataset(**_encoded_train)
	dev_set = TransformersSeqPairDataset(**_encoded_dev)
	test_set = TransformersSeqPairDataset(**_encoded_test)
	logging.info(f"Loaded {len(train_set)} training examples, {len(dev_set)} dev examples and "
				 f"{len(test_set)} test examples")

	train_start = time()
	best_metric, no_increase = float("inf"), 0
	stop_early = False
	for idx_epoch in range(args.num_epochs):
		logging.info(f"Epoch {1 + idx_epoch}/{args.num_epochs}")

		shuffled_indices = torch.randperm(len(train_set))
		train_loss, nb = 0.0, 0

		num_minisets = (len(train_set) + args.validate_every_n_examples - 1) // args.validate_every_n_examples
		for idx_miniset in range(num_minisets):
			logging.info(f"Miniset {1 + idx_miniset}/{num_minisets}")
			curr_subset = Subset(train_set, shuffled_indices[idx_miniset * args.validate_every_n_examples:
															 (idx_miniset + 1) * args.validate_every_n_examples])
			num_subset_batches = (len(curr_subset) + args.batch_size - 1) // args.batch_size

			model.train()
			for curr_batch in tqdm(DataLoader(curr_subset, shuffle=False, batch_size=args.batch_size),
								   total=num_subset_batches):
				res = model(**{k: v.to(DEVICE) for k, v in curr_batch.items()})
				curr_loss = res["loss"]
				train_loss += float(curr_loss)

				curr_loss.backward()
				optimizer.step()
				optimizer.zero_grad()

			nb += num_subset_batches
			logging.info(f"Training loss = {train_loss / nb: .4f}")

			if len(curr_subset) < args.validate_every_n_examples // 2:
				logging.info(f"Skipping validation after training on a small training subset "
							 f"({len(curr_subset)} < {args.validate_every_n_examples // 2} examples)")
				continue

			with torch.no_grad():
				model.eval()
				num_dev_batches = (len(dev_set) + args.batch_size - 1) // args.batch_size

				eval_loss = 0.0
				for curr_batch in tqdm(DataLoader(dev_set, shuffle=False, batch_size=args.batch_size),
									   total=num_dev_batches):
					res = model(**{k: v.to(DEVICE) for k, v in curr_batch.items()})
					eval_loss += float(res["loss"])

				eval_loss /= max(num_dev_batches, 1)

			logging.info(f"Validation loss = {eval_loss: .4f}")

			if eval_loss < best_metric:
				logging.info("New best! Saving checkpoint")
				best_metric = eval_loss
				no_increase = 0
				model.save_pretrained(args.experiment_dir)
			else:
				no_increase += 1

			if no_increase == args.early_stopping_rounds:
				logging.info(f"Stopping early after validation metric did not improve for "
							 f"{args.early_stopping_rounds} rounds.")
				stop_early = True
				break

		if stop_early:
			break

	logging.info(f"Training took {time() - train_start:.4f}s. Best validation metric: {best_metric: .4f}")

	# Reload best model
	model = GPT2LMHeadModel.from_pretrained(args.experiment_dir).to(DEVICE)

	dev_prompts = dev_df["sequence1"].apply(lambda s: f"{s} <PARA>")
	test_prompts = test_df["sequence1"].apply(lambda s: f"{s} <PARA>")

	dev_df["sequence2"].to_csv(os.path.join(args.experiment_dir, "dev_ref.txt"), sep=",", index=False, header=False)
	test_df["sequence2"].to_csv(os.path.join(args.experiment_dir, "test_ref.txt"), sep=",", index=False, header=False)

	strategies = {
		"greedy": {},
		"beam": {"num_beams": 5, "early_stopping": True},
		"top_p": {"do_sample": True, "top_p": 0.9, "top_k": 0},
		"top_k": {"do_sample": True, "top_k": 10}
	}

	with torch.no_grad():
		model.eval()

		for curr_strat, strat_kwargs in strategies.items():
			num_dev_batches = (len(dev_set) + args.batch_size - 1) // args.batch_size

			pred_para = []
			for idx_example in tqdm(range(dev_df.shape[0]), total=dev_df.shape[0]):
				curr_prompt = dev_prompts.iloc[idx_example]
				curr_encoded = tokenizer.batch_encode_plus([curr_prompt], return_tensors="pt")
				take_from_idx = len(curr_encoded["input_ids"][0])
				eff_max_len = len(curr_encoded["input_ids"][0]) + args.max_seq_len

				curr_output = model.generate(curr_encoded["input_ids"].to(DEVICE),
											 pad_token_id=tokenizer.pad_token_id,
											 eos_token_id=tokenizer.eos_token_id,
											 max_length=eff_max_len, **strat_kwargs)
				pred_para.append(tokenizer.decode(curr_output[0, take_from_idx:].cpu(), skip_special_tokens=True))

			with open(os.path.join(args.experiment_dir, f"dev_{curr_strat}_hyp.txt"), "w", encoding="utf-8") as f:
				for _txt in pred_para:
					print(re.sub(r"(\n)+", " ", _txt.strip()), file=f)

			num_test_batches = (len(test_set) + args.batch_size - 1) // args.batch_size
			test_pred_para = []
			for idx_example in tqdm(range(test_df.shape[0]), total=test_df.shape[0]):
				curr_prompt = test_prompts.iloc[idx_example]
				curr_encoded = tokenizer.batch_encode_plus([curr_prompt], return_tensors="pt")
				take_from_idx = len(curr_encoded["input_ids"][0])
				eff_max_len = len(curr_encoded["input_ids"][0]) + args.max_seq_len

				curr_output = model.generate(curr_encoded["input_ids"].to(DEVICE),
											 pad_token_id=tokenizer.pad_token_id,
											 eos_token_id=tokenizer.eos_token_id,
											 max_length=eff_max_len, **strat_kwargs)
				test_pred_para.append(tokenizer.decode(curr_output[0, take_from_idx:].cpu(), skip_special_tokens=True))

			with open(os.path.join(args.experiment_dir, f"test_{curr_strat}_hyp.txt"), "w", encoding="utf-8") as f:
				for _txt in test_pred_para:
					print(re.sub(r"(\n)+", " ", _txt.strip()), file=f)
