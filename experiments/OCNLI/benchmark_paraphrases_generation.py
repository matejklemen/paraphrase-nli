import json
import logging
import os
import sys
from argparse import ArgumentParser
import re

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer

from src.data.cleaning import mask_not_na, inds_unique, mask_long_enough
from src.data.nli import TransformersSeqPairDataset
from src.models.pg_trainer import AutoregressivePGTrainer

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--paraphrase_path", type=str,
					default="home/matej/Documents/paraphrase-nli/experiments/SciTail_NLI/PARAPHRASE_IDENTIFICATION/id-scitail-roberta-base-argmax/all_para_id.csv")
parser.add_argument("--pretrained_name_or_path", type=str, default="ckiplab/gpt2-base-chinese")
parser.add_argument("--model_type", type=str, default="gpt2",
					choices=["gpt2"])

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_seq_len", type=int, default=62)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=5000)
parser.add_argument("--random_seed", type=int, default=17)

parser.add_argument("--use_cpu", action="store_true")

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
		tokenizer_cls = GPT2Tokenizer
	else:
		raise NotImplementedError(f"Model_type '{args.model_type}' is not supported")

	tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)
	tokenizer.add_special_tokens({
		"eos_token": "<EOS>",
		"pad_token": "<PAD>",
		"additional_special_tokens": ["<PARA>"]
	})
	tokenizer.save_pretrained(args.experiment_dir)
	SEPARATOR_ID = int(tokenizer.encode("<PARA>", add_special_tokens=False)[0])

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

	pg_trainer = AutoregressivePGTrainer(args.experiment_dir,
										 pretrained_model_name_or_path=args.pretrained_name_or_path,
										 tokenizer_path=args.experiment_dir,
										 batch_size=args.batch_size,
										 learning_rate=args.learning_rate,
										 validate_every_n_steps=args.validate_every_n_examples,
										 early_stopping_tol=args.early_stopping_rounds,
										 device=("cuda" if not args.use_cpu else "cpu"))
	pg_trainer.run(train_dataset=train_set, val_dataset=dev_set, num_epochs=args.num_epochs)

	# Reload best model
	pg_trainer = AutoregressivePGTrainer.from_pretrained(args.experiment_dir)

	dev_prompts = dev_df["sequence1"].apply(lambda s: f"{s} <PARA>")
	test_prompts = test_df["sequence1"].apply(lambda s: f"{s} <PARA>")

	dev_df["sequence2"].to_csv(os.path.join(args.experiment_dir, "dev_ref.txt"), sep=",", index=False, header=False)
	test_df["sequence2"].to_csv(os.path.join(args.experiment_dir, "test_ref.txt"), sep=",", index=False, header=False)

	dev_df["sequence1"].to_csv(os.path.join(args.experiment_dir, "dev_input_copy.txt"), sep=",", index=False, header=False)
	test_df["sequence1"].to_csv(os.path.join(args.experiment_dir, "test_input_copy.txt"), sep=",", index=False, header=False)

	strategies = {
		"greedy": {},
		"beam": {"num_beams": 5, "early_stopping": True},
		"top_p": {"do_sample": True, "top_p": 0.9, "top_k": 0},
		"top_k": {"do_sample": True, "top_k": 10}
	}

	for curr_strat, strat_kwargs in strategies.items():
		dev_pred_para = pg_trainer.generate(dev_prompts.tolist(), max_seq_len=args.max_seq_len, strategy=strat_kwargs)
		with open(os.path.join(args.experiment_dir, f"dev_{curr_strat}_hyp.txt"), "w", encoding="utf-8") as f:
			for _txt in dev_pred_para:
				print(re.sub(r"(\n)+", " ", _txt.strip()), file=f)

		test_pred_para = pg_trainer.generate(test_prompts.tolist(), max_seq_len=args.max_seq_len, strategy=strat_kwargs)
		with open(os.path.join(args.experiment_dir, f"test_{curr_strat}_hyp.txt"), "w", encoding="utf-8") as f:
			for _txt in test_pred_para:
				print(re.sub(r"(\n)+", " ", _txt.strip()), file=f)
