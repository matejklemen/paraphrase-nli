import json
import logging
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast

from src.data.nli import MultiNLITransformersDataset
from src.models.nli_trainer import TransformersNLITrainer

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--pretrained_name_or_path", type=str, default="bert-base-uncased")
parser.add_argument("--model_type", type=str, default="bert",
                    choices=["bert", "roberta"])

parser.add_argument("--test_path", type=str,
                    default="/home/matej/Documents/data/multinli/multinli_0.9_test_matched_unlabeled.txt")
parser.add_argument("--mode", choices=["matched", "mismatched"], default="matched")

parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--max_seq_len", type=int, default=41)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=100)

parser.add_argument("--use_cpu", action="store_true", default=True)


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
        raise NotImplementedError("Model_type '{args.model_type}' is not supported")

    tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)
    tokenizer.save_pretrained(args.experiment_dir)

    train_set = MultiNLITransformersDataset("train", tokenizer=tokenizer,
                                            max_length=args.max_seq_len, return_tensors="pt")
    dev_set = MultiNLITransformersDataset("validation_matched" if args.mode == "matched" else "validation_mismatched",
                                          tokenizer=tokenizer,
                                          max_length=args.max_seq_len, return_tensors="pt")

    test_set = None
    if args.test_path is not None:
        # Instantiate dev set, override with test data (simplification)
        test_set = MultiNLITransformersDataset("validation_matched[:5]", tokenizer=tokenizer,
                                               max_length=args.max_seq_len, return_tensors="pt")

        df = pd.read_csv(args.test_path, sep="\t")
        encoded = tokenizer.batch_encode_plus(list(zip(df["sentence1"].tolist(), df["sentence2"].tolist())),
                                              max_length=args.max_seq_len, padding="max_length",
                                              truncation="longest_first", return_tensors="pt")
        test_set.str_premise = df["sentence1"].tolist()
        test_set.str_hypothesis = df["sentence2"].tolist()
        test_set.genre = df["genre"].tolist()
        test_set.pair_ids = df["pairID"].tolist()
        for k, v in encoded.items():
            setattr(test_set, k, v)

        test_set.num_examples = len(test_set.str_premise)
        # Labels are from validation set (i.e. invalid)
        delattr(test_set, "labels")
        test_set.valid_attrs.remove("labels")

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
        test_res = trainer.evaluate(test_set)
        if hasattr(test_set, "labels"):
            test_accuracy = float(torch.sum(torch.eq(test_res["pred_label"], test_set.labels))) / len(test_set)
            logging.info(f"Test accuracy: {test_accuracy: .4f}")
        else:
            logging.info(f"Skipping test set evaluation because no labels were found!")

        pd.DataFrame({
            "pairID": test_set.pair_ids,
            "gold_label": list(map(lambda i: test_set.label_names[i], test_res["pred_label"].tolist()))
        }).to_csv(os.path.join(args.experiment_dir, "submission.csv"), sep=",", index=False)
