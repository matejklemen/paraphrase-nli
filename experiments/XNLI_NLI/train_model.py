import csv
import json
import logging
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, CamembertTokenizerFast, \
    AutoTokenizer

from src.data.nli import XNLITransformersDataset
from src.models.nli_trainer import TransformersNLITrainer

import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--lang", type=str, default="de")
parser.add_argument("--en_validation", action="store_true",
                    help="Use English instead of target (--lang) validation set")

parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--pretrained_name_or_path", type=str, default="bert-base-uncased")
parser.add_argument("--model_type", type=str, default="bert",
                    choices=["bert", "camembert", "roberta", "xlm-roberta", "phobert"])

parser.add_argument("--binary_task", action="store_true",
                    help="If set, convert the NLI task into a RTE task, i.e. predicting whether y == entailment or not")
parser.add_argument("--custom_train_path", type=str, default=None,
                    help="If set to a path, will load MNLI train set from this path instead of from 'datasets' library")
parser.add_argument("--custom_dev_path", type=str, default=None,
                    help="If set to a path, will load MNLI dev set from this path instead of from 'datasets' library")
parser.add_argument("--custom_test_path", type=str, default=None,
                    help="If set to a path, will load MNLI test set from this path instead of from 'datasets' library")

parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--max_seq_len", type=int, default=41)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=100)

parser.add_argument("--use_cpu", action="store_true")


if __name__ == "__main__":
    ALL_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    args = parser.parse_args()
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

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
    elif args.model_type == "camembert":
        tokenizer_cls = CamembertTokenizerFast
    elif args.model_type == "roberta":
        tokenizer_cls = RobertaTokenizerFast
    elif args.model_type == "xlm-roberta":
        tokenizer_cls = XLMRobertaTokenizerFast
    else:
        tokenizer_cls = AutoTokenizer

    tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)
    tokenizer.save_pretrained(args.experiment_dir)

    train_set = XNLITransformersDataset("en", "train", tokenizer=tokenizer,
                                        max_length=args.max_seq_len, return_tensors="pt",
                                        binarize=args.binary_task)

    # Override parts with custom (translated) data
    if args.custom_train_path is not None:
        logging.info(f"Loading custom training set from '{args.custom_train_path}'")
        df = pd.read_csv(args.custom_train_path, sep="\t", quoting=csv.QUOTE_NONE)

        is_na = np.logical_or(df["premise"].isna(), df["hypo"].isna())
        logging.info(f"Removing {np.sum(is_na)} sequence pairs due to missing either premise or hypothesis")
        df = df.loc[np.logical_not(is_na)]

        df["label"] = df["label"].apply(lambda lbl: "contradiction" if lbl.lower() == "contradictory" else lbl)
        uniq_labels = set(df["label"])
        assert all([lbl in uniq_labels for lbl in ["entailment", "neutral", "contradiction"]]), \
            f"Non-standard labels: {uniq_labels}"

        encoded = tokenizer.batch_encode_plus(list(zip(df["premise"].tolist(), df["hypo"].tolist())),
                                              max_length=args.max_seq_len, padding="max_length",
                                              truncation="longest_first", return_tensors="pt")
        train_set.str_premise = df["premise"].tolist()
        train_set.str_hypothesis = df["hypo"].tolist()
        if args.binary_task:
            _mapping = {"entailment": 1, "neutral": 0, "contradiction": 0}
            train_set.label_names = ["not_entailment", "entailment"]
            train_set.label2idx = {curr_label: i for i, curr_label in enumerate(train_set.label_names)}
            train_set.idx2label = {i: curr_label for curr_label, i in train_set.label2idx.items()}
        else:
            _mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}

        train_set.labels = torch.tensor(list(map(lambda lbl: _mapping[lbl], df["label"].tolist())))
        for k, v in encoded.items():
            setattr(train_set, k, v)

        train_set.num_examples = len(train_set.str_premise)

    if args.en_validation:
        logging.info(f"Loading English validation set")
        dev_set = XNLITransformersDataset("en", "validation", tokenizer=tokenizer,
                                          max_length=args.max_seq_len, return_tensors="pt",
                                          binarize=args.binary_task)
    else:
        assert args.lang != "all_languages"
        logging.info(f"Loading validation set in language '{args.lang}'")
        dev_set = XNLITransformersDataset(args.lang, "validation", tokenizer=tokenizer,
                                          max_length=args.max_seq_len, return_tensors="pt",
                                          binarize=args.binary_task)
        # Override parts with custom (translated) data
        if args.custom_dev_path is not None:
            logging.info(f"Loading custom validation set from '{args.custom_dev_path}'")
            df_dev = pd.read_csv(args.custom_dev_path, sep="\t")
            uniq_labels = set(df_dev["gold_label"])
            assert all([lbl in uniq_labels for lbl in ["entailment", "neutral", "contradiction"]]), \
                f"Non-standard labels: {uniq_labels}"

            encoded = tokenizer.batch_encode_plus(list(zip(df_dev["sentence1"].tolist(), df_dev["sentence2"].tolist())),
                                                  max_length=args.max_seq_len, padding="max_length",
                                                  truncation="longest_first", return_tensors="pt")
            dev_set.str_premise = df_dev["sentence1"].tolist()
            dev_set.str_hypothesis = df_dev["sentence2"].tolist()
            if args.binary_task:
                _mapping = {"entailment": 1, "neutral": 0, "contradiction": 0}
                dev_set.label_names = ["not_entailment", "entailment"]
                dev_set.label2idx = {curr_label: i for i, curr_label in enumerate(dev_set.label_names)}
                dev_set.idx2label = {i: curr_label for curr_label, i in dev_set.label2idx.items()}
            else:
                _mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}

            dev_set.labels = torch.tensor(list(map(lambda lbl: _mapping[lbl], df_dev["gold_label"].tolist())))
            for k, v in encoded.items():
                setattr(dev_set, k, v)

            dev_set.num_examples = len(dev_set.str_premise)

    # If evaluating on all_languages, handle this in a different way, reloading languages one at a time later on
    loaded_test_lang = args.lang if args.lang != "all_languages" else "en"
    test_set = XNLITransformersDataset(loaded_test_lang, "test", tokenizer=tokenizer,
                                       max_length=args.max_seq_len, return_tensors="pt",
                                       binarize=args.binary_task)
    # Override parts with custom (translated) data
    if args.custom_test_path is not None:
        logging.info(f"Loading custom test set from '{args.custom_test_path}'")
        test_set = XNLITransformersDataset(args.lang, "test", tokenizer=tokenizer,
                                           max_length=args.max_seq_len, return_tensors="pt",
                                           binarize=args.binary_task)

        df_test = pd.read_csv(args.custom_test_path, sep="\t")
        uniq_labels = set(df_test["gold_label"])
        assert all([lbl in uniq_labels for lbl in ["entailment", "neutral", "contradiction"]]), \
            f"Non-standard labels: {uniq_labels}"

        encoded = tokenizer.batch_encode_plus(list(zip(df_test["sentence1"].tolist(), df_test["sentence2"].tolist())),
                                              max_length=args.max_seq_len, padding="max_length",
                                              truncation="longest_first", return_tensors="pt")
        test_set.str_premise = df_test["sentence1"].tolist()
        test_set.str_hypothesis = df_test["sentence2"].tolist()
        if args.binary_task:
            _mapping = {"entailment": 1, "neutral": 0, "contradiction": 0}
            test_set.label_names = ["not_entailment", "entailment"]
            test_set.label2idx = {curr_label: i for i, curr_label in enumerate(test_set.label_names)}
            test_set.idx2label = {i: curr_label for curr_label, i in test_set.label2idx.items()}
        else:
            _mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}

        test_set.labels = torch.tensor(list(map(lambda lbl: _mapping[lbl], df_test["gold_label"].tolist())))
        for k, v in encoded.items():
            setattr(test_set, k, v)

        test_set.num_examples = len(test_set.str_premise)

    logging.info(f"Loaded {len(train_set)} training examples, "
                 f"{len(dev_set)} dev examples and "
                 f"{len(test_set) if test_set is not None else 0} test examples")

    trainer = TransformersNLITrainer(args.experiment_dir,
                                     pretrained_model_name_or_path=args.pretrained_name_or_path,
                                     num_labels=len(train_set.label_names),
                                     batch_size=args.batch_size,
                                     learning_rate=args.learning_rate,
                                     validate_every_n_steps=args.validate_every_n_examples,
                                     early_stopping_tol=args.early_stopping_rounds,
                                     class_weights=([1.0, 2.0] if args.binary_task else None),
                                     optimized_metric=("binary_f1" if args.binary_task else "accuracy"),
                                     device=("cuda" if not args.use_cpu else "cpu"))

    trainer.run(train_dataset=train_set, val_dataset=dev_set, num_epochs=args.num_epochs)

    # Reload best model
    trainer = TransformersNLITrainer.from_pretrained(args.experiment_dir)

    if args.lang == "all_languages":
        dev_set_handles = list(ALL_LANGS)
        test_set_handles = list(ALL_LANGS)
    else:
        # If using custom dev/test (translated) datasets, the datasets are already loaded
        # Otherwise, store the handle and load them later
        dev_set_handles = [(args.lang, dev_set) if args.custom_dev_path is None else args.lang]
        test_set_handles = [(args.lang, test_set) if args.custom_test_path is None else args.lang]

    for curr_handle_or_dataset in dev_set_handles:
        if isinstance(curr_handle_or_dataset, tuple):
            curr_handle, dev_set = curr_handle_or_dataset
        else:
            curr_handle = curr_handle_or_dataset  # type: str
            dev_set = XNLITransformersDataset(curr_handle, "validation", tokenizer=tokenizer,
                                              max_length=args.max_seq_len, return_tensors="pt",
                                              binarize=args.binary_task)
        dev_res = trainer.evaluate(dev_set)

        np_labels = dev_set.labels.numpy()
        np_pred = dev_res["pred_label"].numpy()

        model_metrics = {
            "accuracy": accuracy_score(y_true=np_labels, y_pred=np_pred),
            "macro_precision": precision_score(y_true=np_labels, y_pred=np_pred, average="macro"),
            "macro_recall": recall_score(y_true=np_labels, y_pred=np_pred, average="macro"),
            "macro_f1": f1_score(y_true=np_labels, y_pred=np_pred, average="macro")
        }

        bin_labels = (np_labels == dev_set.label2idx["entailment"]).astype(np.int32)

        for curr_thresh in ["argmax", 0.5, 0.75, 0.9]:
            if curr_thresh == "argmax":
                bin_pred = (np_pred == dev_set.label2idx["entailment"]).astype(np.int32)
            else:
                bin_pred = (dev_res["pred_proba"][:, dev_set.label2idx["entailment"]].numpy() > curr_thresh).astype(np.int32)

            model_metrics[f"thresh-{curr_thresh}"] = {
                "binary_accuracy": accuracy_score(y_true=bin_labels, y_pred=bin_pred),
                "binary_precision": precision_score(y_true=bin_labels, y_pred=bin_pred),
                "binary_recall": recall_score(y_true=bin_labels, y_pred=bin_pred),
                "binary_f1": f1_score(y_true=bin_labels, y_pred=bin_pred)
            }

        logging.info(f"[Dev set, {curr_handle}]\n {model_metrics}")

    if test_set is not None:
        for curr_handle_or_dataset in test_set_handles:
            if isinstance(curr_handle_or_dataset, tuple):
                curr_handle, curr_test_set = curr_handle_or_dataset
            else:
                curr_handle = curr_handle_or_dataset  # type: str
                curr_test_set = XNLITransformersDataset(curr_handle, "test", tokenizer=tokenizer,
                                                        max_length=args.max_seq_len, return_tensors="pt",
                                                        binarize=args.binary_task)

            logging.info(f"Language '{curr_handle}':")
            test_res = trainer.evaluate(curr_test_set)

            np_labels = curr_test_set.labels.numpy()
            np_pred = test_res["pred_label"].numpy()

            conf_matrix = confusion_matrix(y_true=np_labels, y_pred=np_pred)
            plt.matshow(conf_matrix, cmap="Blues")
            for (i, j), v in np.ndenumerate(conf_matrix):
                plt.text(j, i, v, ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            plt.xticks(np.arange(len(test_set.label_names)), test_set.label_names)
            plt.yticks(np.arange(len(test_set.label_names)), test_set.label_names)
            plt.xlabel("(y_pred)")

            plt.savefig(os.path.join(args.experiment_dir, f"{curr_handle}_confusion_matrix.png"))
            logging.info(f"[Test set, {curr_handle}] Confusion matrix:\n {conf_matrix}")

            model_metrics = {
                "accuracy": accuracy_score(y_true=np_labels, y_pred=np_pred),
                "macro_precision": precision_score(y_true=np_labels, y_pred=np_pred, average="macro"),
                "macro_recall": recall_score(y_true=np_labels, y_pred=np_pred, average="macro"),
                "macro_f1": f1_score(y_true=np_labels, y_pred=np_pred, average="macro")
            }

            bin_labels = (np_labels == curr_test_set.label2idx["entailment"]).astype(np.int32)

            for curr_thresh in ["argmax", 0.5, 0.75, 0.9]:
                if curr_thresh == "argmax":
                    bin_pred = (np_pred == curr_test_set.label2idx["entailment"]).astype(np.int32)
                else:
                    bin_pred = (test_res["pred_proba"][:, curr_test_set.label2idx["entailment"]].numpy() > curr_thresh).astype(np.int32)

                conf_matrix = confusion_matrix(y_true=bin_labels, y_pred=bin_pred)
                plt.matshow(conf_matrix, cmap="Blues")
                for (i, j), v in np.ndenumerate(conf_matrix):
                    plt.text(j, i, v, ha='center', va='center',
                             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                plt.xticks([0, 1], ["not_ent", "ent"])
                plt.yticks([0, 1], ["not_ent", "ent"])
                plt.xlabel("(y_pred)")

                plt.savefig(os.path.join(args.experiment_dir, f"{curr_handle}_bin_confusion_matrix_{curr_thresh}.png"))
                logging.info(f"[Test set, {curr_handle}] Confusion matrix, T={curr_thresh}:\n {conf_matrix}")

                model_metrics[f"thresh-{curr_thresh}"] = {
                    "binary_accuracy": accuracy_score(y_true=bin_labels, y_pred=bin_pred),
                    "binary_precision": precision_score(y_true=bin_labels, y_pred=bin_pred),
                    "binary_recall": recall_score(y_true=bin_labels, y_pred=bin_pred),
                    "binary_f1": f1_score(y_true=bin_labels, y_pred=bin_pred)
                }

            with open(os.path.join(args.experiment_dir, f"{curr_handle}_metrics.json"), "w") as f_metrics:
                json.dump(model_metrics, fp=f_metrics, indent=4)

            logging.info(f"[Test set, {curr_handle}]\n {model_metrics}")
