import json
import logging
import os
import sys
from argparse import ArgumentParser
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer

from src.data import TransformersSeqPairDataset
from src.models.nli_trainer import TransformersNLITrainer


class COPARelVerTransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: str, tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False):
        data = pd.read_csv(path, sep="\t")
        data = data.loc[np.logical_not(np.logical_or(data["sentence1"].isna(), data["sentence2"].isna()))]

        self.seq1 = data["sentence1"].tolist()
        self.seq2 = data["sentence2"].tolist()
        valid_label = data["is_gennli"].tolist()

        if reverse_order:
            self.seq1, self.seq2 = self.seq2, self.seq1

        self.label_names = ["not_gennli", "gennli"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Sanity check: expecting binary task
        assert len(set(valid_label)) == 2

        optional_kwargs = {}
        if return_tensors is not None:
            valid_label = torch.tensor(valid_label)
            optional_kwargs["return_tensors"] = "pt"

        if max_length is not None:
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = tokenizer.batch_encode_plus(list(zip(self.seq1, self.seq2)), **optional_kwargs)
        encoded["labels"] = valid_label

        super().__init__(**encoded)

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--pretrained_name_or_path", type=str, default="roberta-base")

parser.add_argument("--reverse_order", action="store_true")
parser.add_argument("--optimized_metric", default="accuracy",
                    choices=["loss", "accuracy", "binary_f1"])

parser.add_argument("--train_path", type=str, help="Path to training set of COPA_relver (tsv)",
                    default="id_copa_genent_train.tsv")
parser.add_argument("--dev_path", type=str, help="Path to dev set of COPA_relver (tsv)",
                    default="id_copa_genent_dev.tsv")
parser.add_argument("--test_path", type=str, help="Path to test set of COPA_relver (tsv)",
                    default="id_copa_genent_test.tsv")

parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--max_seq_len", type=int, default=22)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=500)

parser.add_argument("--use_cpu", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not torch.cuda.is_available() and not args.use_cpu:
        args.use_cpu = True

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

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    tokenizer.save_pretrained(args.experiment_dir)

    train_set = COPARelVerTransformersDataset(args.train_path, tokenizer=tokenizer,
                                              max_length=args.max_seq_len, return_tensors="pt",
                                              reverse_order=args.reverse_order)
    dev_set = COPARelVerTransformersDataset(args.dev_path, tokenizer=tokenizer,
                                            max_length=args.max_seq_len, return_tensors="pt",
                                            reverse_order=args.reverse_order)
    test_set = COPARelVerTransformersDataset(args.test_path, tokenizer=tokenizer,
                                             max_length=args.max_seq_len, return_tensors="pt",
                                             reverse_order=args.reverse_order)

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
                                     optimized_metric=args.optimized_metric,
                                     device=("cuda" if not args.use_cpu else "cpu"))

    trainer.run(train_dataset=train_set, val_dataset=dev_set, num_epochs=args.num_epochs)

    trainer = TransformersNLITrainer.from_pretrained(args.experiment_dir)
    test_res = trainer.evaluate(test_set)

    np_labels = test_set.labels.numpy()

    model_metrics = {}
    # Warning: threshold is specified for "not_translation" label, but pred=1 indicates a translation!
    for curr_thresh in ["argmax", 0.75, 0.9]:
        if curr_thresh == "argmax":
            curr_pred = test_res["pred_label"].numpy()
        else:
            curr_pred = np.logical_not(
                test_res["pred_proba"][:, test_set.label2idx["not_gennli"]].numpy() > curr_thresh
            ).astype(np.int32)

        conf_matrix = confusion_matrix(y_true=np_labels, y_pred=curr_pred)
        plt.matshow(conf_matrix, cmap="Blues")
        for (i, j), v in np.ndenumerate(conf_matrix):
            plt.text(j, i, v, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.xticks([0, 1], test_set.label_names)
        plt.yticks([0, 1], test_set.label_names)
        plt.xlabel("(y_pred)")

        plt.savefig(os.path.join(args.experiment_dir, f"confusion_matrix_{curr_thresh}.png"))
        logging.info(f"Confusion matrix ({curr_thresh}):\n {conf_matrix}")

        model_metrics[f"thresh-{curr_thresh}"] = {
            "binary_accuracy": accuracy_score(y_true=np_labels, y_pred=curr_pred),
            "binary_precision": precision_score(y_true=np_labels, y_pred=curr_pred,
                                                average="binary", pos_label=0),
            "binary_recall": recall_score(y_true=np_labels, y_pred=curr_pred,
                                          average="binary", pos_label=0),
            "binary_f1": f1_score(y_true=np_labels, y_pred=curr_pred,
                                  average="binary", pos_label=0)
        }

    with open(os.path.join(args.experiment_dir, "metrics.json"), "w") as f_metrics:
        logging.info(model_metrics)
        json.dump(model_metrics, fp=f_metrics, indent=4)

    logging.info(model_metrics)
