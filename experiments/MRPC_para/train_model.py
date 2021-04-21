import json
import logging
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast

from src.data.paraphrase import MRPCTransformersDataset
from src.models.nli_trainer import TransformersNLITrainer

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--pretrained_name_or_path", type=str, default="roberta-base")
parser.add_argument("--model_type", type=str, default="roberta",
                    choices=["bert", "roberta", "xlm-roberta"])

parser.add_argument("--balance", action="store_true")
parser.add_argument("--reverse_order", action="store_true")
parser.add_argument("--create_dev_from_training", action="store_true",
                    help="If set, split the training set use one half as a test set substitute")
parser.add_argument("--optimized_metric", default="binary_f1",
                    choices=["loss", "accuracy", "binary_f1"])

parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/mrpc/msr_paraphrase_train.txt")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/mrpc/msr_paraphrase_test.txt")

parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--max_seq_len", type=int, default=74)
parser.add_argument("--batch_size", type=int, default=16)
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

    train_set = MRPCTransformersDataset(args.train_path, tokenizer=tokenizer,
                                        max_length=args.max_seq_len, return_tensors="pt",
                                        reverse_order=args.reverse_order, balance=args.balance)
    test_set = MRPCTransformersDataset(args.test_path, tokenizer=tokenizer,
                                       max_length=args.max_seq_len, return_tensors="pt",
                                       reverse_order=args.reverse_order)

    if args.create_dev_from_training:
        indices = torch.randperm(len(train_set)).tolist()
        dev_indices = indices[-(500 if args.balance else 1000):]
        train_indices = indices[:-(500 if args.balance else 1000)]

        dev_set = deepcopy(train_set)
        # NOTE: order important here! First copy to test set, then fix dev set itself
        for curr_set, curr_indices in [(dev_set, dev_indices), (train_set, train_indices)]:
            curr_set.sid1 = [train_set.sid1[_i] for _i in curr_indices]
            curr_set.sid2 = [train_set.sid2[_i] for _i in curr_indices]

            curr_set.seq1 = [train_set.seq1[_i] for _i in curr_indices]
            curr_set.seq2 = [train_set.seq2[_i] for _i in curr_indices]

            for attr in train_set.valid_attrs:
                setattr(curr_set, attr, getattr(train_set, attr)[curr_indices])

            curr_set.num_examples = len(curr_set.sid1)
    else:
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
                                     class_weights=(None if args.balance else [2.0, 1.0]),
                                     optimized_metric=args.optimized_metric,
                                     device=("cuda" if not args.use_cpu else "cpu"))

    trainer.run(train_dataset=train_set, val_dataset=dev_set, num_epochs=args.num_epochs)

    if test_set is not None:
        trainer = TransformersNLITrainer.from_pretrained(args.experiment_dir)
        test_res = trainer.evaluate(test_set)

        if hasattr(test_set, "labels"):
            np_labels = test_set.labels.numpy()

            model_metrics = {}
            # Warning: threshold is specified for "not_paraphrase" label, but pred=1 indicates a paraphrase!
            for curr_thresh in ["argmax", 0.75, 0.9]:
                if curr_thresh == "argmax":
                    curr_pred = test_res["pred_label"].numpy()
                else:
                    curr_pred = np.logical_not(
                        test_res["pred_proba"][:, test_set.label2idx["not_paraphrase"]].numpy() > curr_thresh
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
