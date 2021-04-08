import csv
import json
import logging
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, CamembertTokenizerFast, \
    AutoTokenizer

from src.data.nli import XNLITransformersDataset
from src.models.nli_trainer import TransformersNLITrainer

parser = ArgumentParser()
parser.add_argument("--lang", type=str, default="de")
parser.add_argument("--en_validation", action="store_true",
                    help="Use English instead of target (--lang) validation set")
parser.add_argument("--bilingual_validation", action="store_true",
                    help="Use combined English and target (--lang) validation set")

parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--pretrained_name_or_path", type=str, default="bert-base-uncased")
parser.add_argument("--model_type", type=str, default="bert",
                    choices=["bert", "camembert", "roberta", "xlm-roberta", "phobert"])

parser.add_argument("--custom_train_path", type=str, default=None,
                    help="If set to a path, will load MNLI train set from this path instead of from 'datasets' library")
parser.add_argument("--custom_dev_path", type=str, default=None,
                    help="If set to a path, will load MNLI dev set from this path instead of from 'datasets' library")
parser.add_argument("--custom_test_path", type=str, default=None,
                    help="If set to a path, will load MNLI test set from this path instead of from 'datasets' library")
parser.add_argument("--combine_train_dev", action="store_true")

parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--max_seq_len", type=int, default=41)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=100)

parser.add_argument("--use_cpu", action="store_true")


if __name__ == "__main__":
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
                                        max_length=args.max_seq_len, return_tensors="pt")

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
        train_set.labels = torch.tensor(list(map(lambda lbl: train_set.label2idx[lbl], df["label"].tolist())))
        for k, v in encoded.items():
            setattr(train_set, k, v)

        train_set.num_examples = len(train_set.str_premise)

    if args.en_validation:
        dev_set = XNLITransformersDataset("en", "validation", tokenizer=tokenizer,
                                          max_length=args.max_seq_len, return_tensors="pt")
    elif args.bilingual_validation:
        dev_set = XNLITransformersDataset(("en", args.lang), ("validation", "validation"), tokenizer=tokenizer,
                                          max_length=args.max_seq_len, return_tensors="pt")
    else:
        dev_set = XNLITransformersDataset(args.lang, "validation", tokenizer=tokenizer,
                                          max_length=args.max_seq_len, return_tensors="pt")
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
            dev_set.labels = torch.tensor(list(map(lambda lbl: dev_set.label2idx[lbl], df_dev["gold_label"].tolist())))
            for k, v in encoded.items():
                setattr(dev_set, k, v)

            dev_set.num_examples = len(dev_set.str_premise)

    test_set = XNLITransformersDataset(args.lang, "test", tokenizer=tokenizer,
                                       max_length=args.max_seq_len, return_tensors="pt")
    # Override parts with custom (translated) data
    if args.custom_test_path is not None:
        logging.info(f"Loading custom test set from '{args.custom_test_path}'")
        test_set = XNLITransformersDataset(args.lang, "test", tokenizer=tokenizer,
                                           max_length=args.max_seq_len, return_tensors="pt")

        df_test = pd.read_csv(args.custom_test_path, sep="\t")
        uniq_labels = set(df_test["gold_label"])
        assert all([lbl in uniq_labels for lbl in ["entailment", "neutral", "contradiction"]]), \
            f"Non-standard labels: {uniq_labels}"

        encoded = tokenizer.batch_encode_plus(list(zip(df_test["sentence1"].tolist(), df_test["sentence2"].tolist())),
                                              max_length=args.max_seq_len, padding="max_length",
                                              truncation="longest_first", return_tensors="pt")
        test_set.str_premise = df_test["sentence1"].tolist()
        test_set.str_hypothesis = df_test["sentence2"].tolist()
        test_set.labels = torch.tensor(list(map(lambda lbl: test_set.label2idx[lbl], df_test["gold_label"].tolist())))
        for k, v in encoded.items():
            setattr(test_set, k, v)

        test_set.num_examples = len(test_set.str_premise)

    if args.combine_train_dev:
        train_set.str_premise = train_set.str_premise + dev_set.str_premise
        train_set.str_hypothesis = train_set.str_hypothesis + dev_set.str_hypothesis

        for k in train_set.valid_attrs:
            train_k = getattr(train_set, k)
            dev_k = getattr(dev_set, k)
            merged = torch.cat((train_k, dev_k))
            logging.info(f"Merging together '{k}' for training ({train_k.shape}) and dev set ({dev_k.shape}): "
                         f"merged shape = {merged.shape}")
            setattr(train_set, k, merged)

        train_set.num_examples = train_set.num_examples + dev_set.num_examples

        dev_set = test_set
        test_set = None

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
                                     device=("cuda" if not args.use_cpu else "cpu"))

    trainer.run(train_dataset=train_set, val_dataset=dev_set, num_epochs=args.num_epochs)

    if test_set is not None:
        trainer = TransformersNLITrainer.from_pretrained(args.experiment_dir)
        # If using English-only or bilingual dev set, obtain a score on monolingual dev set to aid comparison
        if args.en_validation or args.bilingual_validation:
            dev_set = XNLITransformersDataset(args.lang, "validation", tokenizer=tokenizer,
                                              max_length=args.max_seq_len, return_tensors="pt")
            dev_res = trainer.evaluate(dev_set)
            dev_accuracy = float(torch.sum(torch.eq(dev_res["pred_label"], dev_set.labels))) / len(dev_set)
            logging.info(f"Dev accuracy ('{args.lang}' only): {dev_accuracy: .4f}")

        test_res = trainer.evaluate(test_set)
        if hasattr(test_set, "labels"):
            test_accuracy = float(torch.sum(torch.eq(test_res["pred_label"], test_set.labels))) / len(test_set)
            logging.info(f"Test accuracy: {test_accuracy: .4f}")
        else:
            logging.info(f"Skipping test set evaluation because no labels were found!")
