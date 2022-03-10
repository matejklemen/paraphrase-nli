import json
import logging
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast

from src.data.cleaning import mask_not_na, inds_unique, mask_long_enough
from src.data.nli import TransformersSeqPairDataset
from src.models.nli_trainer import TransformersPITrainer

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--paraphrase_path", type=str,
					default="/home/matej/Documents/paraphrase-nli/experiments/SNLI_NLI/PARAPHRASE_IDENTIFICATION/snli-bin-roberta-base-argmax/all_para_id.csv")
parser.add_argument("--pretrained_name_or_path", type=str, default="roberta-base")
parser.add_argument("--model_type", type=str, default="roberta",
					choices=["bert", "roberta", "xlm-roberta"])

parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--max_seq_len", type=int, default=41)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=100)
parser.add_argument("--random_seed", type=int, default=17)

parser.add_argument("--use_cpu", action="store_true")

if __name__ == "__main__":
	USED_LABELSET = ["not_paraphrase", "paraphrase"]
	args = parser.parse_args()
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
	if args.model_type == "bert":
		tokenizer_cls = BertTokenizerFast
	elif args.model_type == "roberta":
		tokenizer_cls = RobertaTokenizerFast
	elif args.model_type == "xlm-roberta":
		tokenizer_cls = XLMRobertaTokenizerFast
	else:
		raise NotImplementedError(f"Model_type '{args.model_type}' is not supported")

	tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)
	tokenizer.save_pretrained(args.experiment_dir)

	df = pd.read_csv(args.paraphrase_path)
	# Basic data cleaning - remove NAs (?), duplicate pairs, pairs with one sequence very short
	df = df.loc[mask_not_na(df["sequence1"], df["sequence2"])]
	df = df.iloc[inds_unique(df["sequence1"], df["sequence2"])]
	df = df.loc[mask_long_enough(df["sequence1"], df["sequence2"])]

	# Balance dataset by undersampling majority class
	uniq_classes, uniq_counts = np.unique(df["label"].values, return_counts=True)
	minority, majority = np.argmin(uniq_counts), np.argmax(uniq_counts)
	minority_count, majority_count = uniq_counts[minority], uniq_counts[majority]

	balanced_data = pd.concat((
		df.loc[df["label"] == minority].reset_index(drop=True),
		df.loc[df["label"] == majority].sample(n=minority_count).reset_index(drop=True)
	))
	num_ex = balanced_data.shape[0]

	indices = np.random.permutation(num_ex)
	train_df = balanced_data.iloc[indices[:int(0.8 * num_ex)]]
	dev_df = balanced_data.iloc[indices[int(0.8 * num_ex): int(0.9 * num_ex)]]
	test_df = balanced_data.iloc[indices[int(0.9 * num_ex):]]

	tokenizer_cls = RobertaTokenizerFast
	tokenizer = tokenizer_cls.from_pretrained("roberta-base")

	_encoded_train = tokenizer.batch_encode_plus(
		list(zip(train_df["sequence1"].tolist(), train_df["sequence2"].tolist())),
		max_length=args.max_seq_len, padding="max_length", truncation="longest_first", return_tensors="pt"
	)
	_encoded_train["labels"] = torch.tensor(train_df["label"].tolist())

	_encoded_dev = tokenizer.batch_encode_plus(
		list(zip(dev_df["sequence1"].tolist(), dev_df["sequence2"].tolist())),
		max_length=args.max_seq_len, padding="max_length", truncation="longest_first", return_tensors="pt"
	)
	_encoded_dev["labels"] = torch.tensor(dev_df["label"].tolist())

	_encoded_test = tokenizer.batch_encode_plus(
		list(zip(test_df["sequence1"].tolist(), test_df["sequence2"].tolist())),
		max_length=args.max_seq_len, padding="max_length", truncation="longest_first", return_tensors="pt"
	)
	_encoded_test["labels"] = torch.tensor(test_df["label"].tolist())

	train_set = TransformersSeqPairDataset(**_encoded_train)
	dev_set = TransformersSeqPairDataset(**_encoded_dev)
	test_set = TransformersSeqPairDataset(**_encoded_test)
	logging.info(f"Loaded {len(train_set)} training examples, "
				 f"{len(dev_set)} dev examples and "
				 f"{len(test_set) if test_set is not None else 0} test examples")

	trainer = TransformersPITrainer(args.experiment_dir,
									pretrained_model_name_or_path=args.pretrained_name_or_path,
									num_labels=len(train_set.label_names),
									batch_size=args.batch_size,
									learning_rate=args.learning_rate,
									validate_every_n_steps=args.validate_every_n_examples,
									early_stopping_tol=args.early_stopping_rounds,
									optimized_metric="accuracy",
									device=("cuda" if not args.use_cpu else "cpu"))
	trainer.run(train_dataset=train_set, val_dataset=dev_set, num_epochs=args.num_epochs)

	trainer = TransformersPITrainer.from_pretrained(args.experiment_dir)
	test_res = trainer.evaluate(test_set)

	if hasattr(test_set, "labels"):
		np_labels = test_set.labels.numpy()
		np_pred = test_res["pred_label"].numpy()

		conf_matrix = confusion_matrix(y_true=np_labels, y_pred=np_pred)
		plt.matshow(conf_matrix, cmap="Blues")
		for (i, j), v in np.ndenumerate(conf_matrix):
			plt.text(j, i, v, ha='center', va='center',
					 bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
		plt.xticks(np.arange(len(test_set.label_names)), USED_LABELSET)
		plt.yticks(np.arange(len(test_set.label_names)), USED_LABELSET)
		plt.xlabel("(y_pred)")

		plt.savefig(os.path.join(args.experiment_dir, "confusion_matrix.png"))
		logging.info(f"Confusion matrix:\n {conf_matrix}")

		model_metrics = {
			"accuracy": accuracy_score(y_true=np_labels, y_pred=np_pred),
			"macro_precision": precision_score(y_true=np_labels, y_pred=np_pred, average="macro"),
			"macro_recall": recall_score(y_true=np_labels, y_pred=np_pred, average="macro"),
			"macro_f1": f1_score(y_true=np_labels, y_pred=np_pred, average="macro")
		}

		bin_labels = (np_labels == 1).astype(np.int32)

		for curr_thresh in ["argmax", 0.5, 0.75, 0.9]:
			if curr_thresh == "argmax":
				bin_pred = (np_pred == 1).astype(np.int32)
			else:
				bin_pred = (test_res["pred_proba"][:, 1].numpy() > curr_thresh).astype(np.int32)

			conf_matrix = confusion_matrix(y_true=bin_labels, y_pred=bin_pred)
			plt.matshow(conf_matrix, cmap="Blues")
			for (i, j), v in np.ndenumerate(conf_matrix):
				plt.text(j, i, v, ha='center', va='center',
						 bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
			plt.xticks([0, 1], USED_LABELSET)
			plt.yticks([0, 1], USED_LABELSET)
			plt.xlabel("(y_pred)")

			plt.savefig(os.path.join(args.experiment_dir, f"bin_confusion_matrix_{curr_thresh}.png"))
			logging.info(f"Confusion matrix ({curr_thresh}):\n {conf_matrix}")

			model_metrics[f"thresh-{curr_thresh}"] = {
				"binary_accuracy": accuracy_score(y_true=bin_labels, y_pred=bin_pred),
				"binary_precision": precision_score(y_true=bin_labels, y_pred=bin_pred),
				"binary_recall": recall_score(y_true=bin_labels, y_pred=bin_pred),
				"binary_f1": f1_score(y_true=bin_labels, y_pred=bin_pred)
			}

		with open(os.path.join(args.experiment_dir, "metrics.json"), "w") as f_metrics:
			logging.info(model_metrics)
			json.dump(model_metrics, fp=f_metrics, indent=4)

		logging.info(model_metrics)
