import json
import logging
import os
import sys
from argparse import ArgumentParser

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast

from src.data.nli import SciTailTransformersDataset
from src.models.nli_trainer import TransformersNLITrainer

import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--pretrained_name_or_path", type=str, default="bert-base-uncased")
parser.add_argument("--model_type", type=str, default="bert",
                    choices=["bert", "roberta", "xlm-roberta"])

parser.add_argument("--combine_train_dev", action="store_true")

parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--max_seq_len", type=int, default=66)
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
    elif args.model_type == "roberta":
        tokenizer_cls = RobertaTokenizerFast
    elif args.model_type == "xlm-roberta":
        tokenizer_cls = XLMRobertaTokenizerFast
    else:
        raise NotImplementedError(f"Model_type '{args.model_type}' is not supported")

    tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)
    tokenizer.save_pretrained(args.experiment_dir)

    test_set = None
    if args.combine_train_dev:
        train_set = SciTailTransformersDataset(("train", "validation"), tokenizer=tokenizer,
                                               max_length=args.max_seq_len, return_tensors="pt")
        dev_set = SciTailTransformersDataset("test", tokenizer=tokenizer,
                                             max_length=args.max_seq_len, return_tensors="pt")
    else:
        train_set = SciTailTransformersDataset("train", tokenizer=tokenizer,
                                               max_length=args.max_seq_len, return_tensors="pt")
        dev_set = SciTailTransformersDataset("validation", tokenizer=tokenizer,
                                             max_length=args.max_seq_len, return_tensors="pt")
        test_set = SciTailTransformersDataset("test", tokenizer=tokenizer,
                                              max_length=args.max_seq_len, return_tensors="pt")

    logging.info(f"Loaded {len(train_set)} training examples, "
                 f"{len(dev_set)} dev examples and "
                 f"{len(test_set) if test_set is not None else 0} test examples")

    uniq_labels, label_counts = np.unique(train_set.labels, return_counts=True)
    # There are more neutral (0) examples than entails (1), provide ratio as a weight for cross-entropy
    label_ratio = label_counts[0] / label_counts[1]
    assert label_ratio >= 1.0

    trainer = TransformersNLITrainer(args.experiment_dir,
                                     pretrained_model_name_or_path=args.pretrained_name_or_path,
                                     num_labels=len(train_set.label_names),
                                     batch_size=args.batch_size,
                                     learning_rate=args.learning_rate,
                                     validate_every_n_steps=args.validate_every_n_examples,
                                     early_stopping_tol=args.early_stopping_rounds,
                                     class_weights=[1.0, label_ratio],
                                     optimized_metric="binary_f1",
                                     device=("cuda" if not args.use_cpu else "cpu"))

    trainer.run(train_dataset=train_set, val_dataset=dev_set, num_epochs=args.num_epochs)

    if test_set is not None:
        trainer = TransformersNLITrainer.from_pretrained(args.experiment_dir)
        test_res = trainer.evaluate(test_set)
        if hasattr(test_set, "labels"):
            np_labels = test_set.labels.numpy()
            np_pred = test_res["pred_label"].numpy()

            conf_matrix = confusion_matrix(y_true=np_labels, y_pred=np_pred)
            plt.matshow(conf_matrix, cmap="Blues")
            for (i, j), v in np.ndenumerate(conf_matrix):
                plt.text(j, i, v, ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            plt.xticks(np.arange(len(test_set.label_names)), test_set.label_names)
            plt.yticks(np.arange(len(test_set.label_names)), test_set.label_names)
            plt.xlabel("(y_pred)")

            plt.savefig(os.path.join(args.experiment_dir, "confusion_matrix.png"))
            logging.info(f"Confusion matrix:\n {conf_matrix}")

            model_metrics = {
                "accuracy": accuracy_score(y_true=np_labels, y_pred=np_pred),
                "macro_precision": precision_score(y_true=np_labels, y_pred=np_pred, average="macro"),
                "macro_recall": recall_score(y_true=np_labels, y_pred=np_pred, average="macro"),
                "macro_f1": f1_score(y_true=np_labels, y_pred=np_pred, average="macro")
            }

            bin_labels = (np_labels == test_set.label2idx["entails"]).astype(np.int32)

            for curr_thresh in ["argmax", 0.5, 0.75, 0.9]:
                if curr_thresh == "argmax":
                    bin_pred = (np_pred == test_set.label2idx["entails"]).astype(np.int32)
                else:
                    bin_pred = (test_res["pred_proba"][:, test_set.label2idx["entails"]].numpy() > curr_thresh).astype(np.int32)

                conf_matrix = confusion_matrix(y_true=bin_labels, y_pred=bin_pred)
                plt.matshow(conf_matrix, cmap="Blues")
                for (i, j), v in np.ndenumerate(conf_matrix):
                    plt.text(j, i, v, ha='center', va='center',
                             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                plt.xticks([0, 1], ["not_ent", "ent"])
                plt.yticks([0, 1], ["not_ent", "ent"])
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
        else:
            logging.info(f"Skipping test set evaluation because no labels were found!")
