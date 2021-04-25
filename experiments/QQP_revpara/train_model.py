import argparse
import csv
import json
import logging
import os
import sys

from io import StringIO
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/qqp/train.tsv")
parser.add_argument("--dev_path", type=str, default="/home/matej/Documents/data/qqp/dev.tsv")
parser.add_argument("--pretrained_name_or_path", type=str, default="roberta-base")

parser.add_argument("--label_embedding_size", type=int, default=50)
parser.add_argument("--fc_hidden_size", type=int, default=50)

parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_seq_len", type=int, default=55)
parser.add_argument("--validate_every_n_steps", type=int, default=5000)
parser.add_argument("--early_stopping_rounds", type=int, default=5)


def read_qqp(file_path):
    data_dict = {
        "pair_id": [],
        "qid1": [],
        "qid2": [],
        "seq1": [],
        "seq2": [],
        "label": []
    }

    with open(file_path, "r", encoding="utf-8") as f:
        data = list(map(lambda s: s.strip(), f.readlines()))

    header = data[0].split("\t")
    data_dict["header"] = header
    assert ("is_duplicate" in header and len(header) == 6) or \
           ("is_duplicate" not in header and len(header) == 5)

    num_errs = 0
    for i, curr_row in enumerate(data[1:], start=1):
        fields = list(csv.reader(StringIO(curr_row), delimiter="\t"))[0]

        if len(fields) != len(header):
            num_errs += 1
            continue

        data_dict["pair_id"].append(int(fields[0]))
        data_dict["qid1"].append(int(fields[1]))
        data_dict["qid2"].append(int(fields[2]))
        data_dict["seq1"].append(fields[3])
        data_dict["seq2"].append(fields[4])

        if len(fields) == 6:
            data_dict["label"].append(int(fields[-1]))

    return data_dict


class DecomposablePairModel(nn.Module):
    def __init__(self, transformers_handle, label_embedding_size: int = 50, num_labels: int = 2,
                 fc_hidden_size: int = 10, class_weights: Optional = None):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(transformers_handle, return_dict=True,
                                                                        num_labels=3)  # "motivation": entailment/neutral/contradiction
        self.label_embedder = nn.Linear(in_features=self.model.config.num_labels,
                                        out_features=label_embedding_size)

        self.fc = nn.Sequential(
            nn.Linear(in_features=(2 * label_embedding_size), out_features=fc_hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=fc_hidden_size, out_features=num_labels)
        )

        self.class_weights = class_weights
        self.num_labels = num_labels

    def forward(self, forward_ids, backward_ids, **modeling_kwargs):
        bert_res = self.model(forward_ids, **modeling_kwargs.get("forward_kwargs", {}))
        forward_hidden = torch.softmax(bert_res["logits"], dim=-1)  # [B, 3]
        forward_emb = self.label_embedder(forward_hidden)  # [B, label_embedding_size]

        bert_res = self.model(backward_ids, **modeling_kwargs.get("backward_kwargs", {}))
        back_hidden = torch.softmax(bert_res["logits"], dim=-1)  # [B, 3]
        back_emb = self.label_embedder(back_hidden)  # [B, label_embedding_size]

        combined = torch.cat((forward_emb, back_emb), dim=-1)  # [B, 2*label_embedding_size]
        logits = self.fc(combined)
        ret_dict = {"logits": logits}

        if "label" in modeling_kwargs:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(logits.view(-1, self.num_labels), modeling_kwargs["label"].view(-1))
            ret_dict["loss"] = loss

        return ret_dict


if __name__ == "__main__":
    from transformers import RobertaTokenizerFast
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = parser.parse_args()
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

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

    with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), fp=f, indent=4)

    train_data = read_qqp(args.train_path)
    dev_data = read_qqp(args.dev_path)

    random_indices = torch.randperm(len(dev_data["label"]))
    dev_indices = random_indices[: int(0.5 * len(dev_data["label"]))]
    test_indices = random_indices[int(0.5 * len(dev_data["label"])):]

    with open(os.path.join(args.experiment_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump({
           "dev_indices": dev_indices.tolist(),
           "test_indices": test_indices.tolist()
        }, fp=f, indent=4)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.pretrained_name_or_path)
    tokenizer.save_pretrained(args.experiment_dir)

    train_encoded = tokenizer.batch_encode_plus(list(zip(train_data["seq1"], train_data["seq2"])),
                                                max_length=args.max_seq_len,
                                                padding="max_length", truncation="longest_first", return_tensors="pt")
    train_rev_encoded = tokenizer.batch_encode_plus(list(zip(train_data["seq2"], train_data["seq1"])),
                                                    max_length=args.max_seq_len,
                                                    padding="max_length", truncation="longest_first", return_tensors="pt")
    train_labels = torch.tensor(train_data["label"])

    dev_encoded = tokenizer.batch_encode_plus([(dev_data["seq1"][_i], dev_data["seq2"][_i]) for _i in dev_indices],
                                              max_length=args.max_seq_len,
                                              padding="max_length", truncation="longest_first", return_tensors="pt")
    dev_rev_encoded = tokenizer.batch_encode_plus([(dev_data["seq2"][_i], dev_data["seq1"][_i]) for _i in dev_indices],
                                                  max_length=args.max_seq_len,
                                                  padding="max_length", truncation="longest_first", return_tensors="pt")
    dev_labels = torch.tensor([dev_data["label"][_i] for _i in dev_indices])

    test_encoded = tokenizer.batch_encode_plus([(dev_data["seq1"][_i], dev_data["seq2"][_i]) for _i in test_indices],
                                               max_length=args.max_seq_len,
                                               padding="max_length", truncation="longest_first", return_tensors="pt")
    test_rev_encoded = tokenizer.batch_encode_plus([(dev_data["seq2"][_i], dev_data["seq1"][_i]) for _i in test_indices],
                                                   max_length=args.max_seq_len,
                                                   padding="max_length", truncation="longest_first", return_tensors="pt")
    test_seq1 = [dev_data["seq1"][_i] for _i in test_indices]
    test_seq2 = [dev_data["seq2"][_i] for _i in test_indices]
    test_labels = torch.tensor([dev_data["label"][_i] for _i in test_indices])

    logging.info(f"Loaded "
                 f"{len(train_encoded['input_ids'])} train examples,"
                 f"{len(dev_encoded['input_ids'])} dev examples,"
                 f"{len(test_encoded['input_ids'])} test examples")

    model = DecomposablePairModel(transformers_handle=args.pretrained_name_or_path,
                                  label_embedding_size=args.label_embedding_size,
                                  fc_hidden_size=args.fc_hidden_size,
                                  class_weights=torch.tensor([1.0, 2.0], device=DEVICE),
                                  num_labels=len(set(train_data["label"]))).to(DEVICE)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    VALIDATE_EVERY_N_STEPS = args.validate_every_n_steps
    BATCH_SIZE = args.batch_size
    DEV_BATCH_SIZE = 3 * BATCH_SIZE
    num_subsets = (len(train_encoded["input_ids"]) + VALIDATE_EVERY_N_STEPS - 1) // VALIDATE_EVERY_N_STEPS
    num_dev_batches = (len(dev_encoded["input_ids"]) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE

    best_f1, no_increase = -float("inf"), 0

    for idx_epoch in range(args.num_epochs):
        logging.info(f"Epoch#{idx_epoch}/{args.num_epochs}")
        train_loss, train_denom = 0.0, 0

        for idx_subset in range(num_subsets):
            model.train()

            ssub, esub = idx_subset * VALIDATE_EVERY_N_STEPS, (idx_subset + 1) * VALIDATE_EVERY_N_STEPS
            subset_encoded = {k: train_encoded[k][ssub: esub] for k in train_encoded.keys()}
            subset_rev_encoded = {k: train_rev_encoded[k][ssub: esub] for k in train_encoded.keys()}
            subset_labels = train_labels[ssub: esub]

            num_train_batches = (len(subset_encoded["input_ids"]) + BATCH_SIZE - 1) // BATCH_SIZE

            for idx_batch in tqdm(range(num_train_batches), total=num_train_batches):
                sb, eb = idx_batch * BATCH_SIZE, (idx_batch + 1) * BATCH_SIZE

                res = model(forward_ids=subset_encoded["input_ids"][sb: eb].to(DEVICE),
                            backward_ids=subset_rev_encoded["input_ids"][sb: eb].to(DEVICE),
                            forward_kwargs={k: subset_encoded[k][sb: eb].to(DEVICE) for k in ["attention_mask"]},
                            backward_kwargs={k: subset_rev_encoded[k][sb: eb].to(DEVICE) for k in ["attention_mask"]},
                            label=subset_labels[sb: eb].to(DEVICE))

                loss = res["loss"]
                train_loss += float(loss)
                train_denom += 1

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            logging.info(f"Training loss: {train_loss / max(1, train_denom): .4f}")

            dev_loss, dev_denom = 0.0, 0
            with torch.no_grad():
                model.eval()
                preds = []

                for idx_dev_batch in tqdm(range(num_dev_batches), total=num_dev_batches):
                    sb, eb = idx_dev_batch * DEV_BATCH_SIZE, (idx_dev_batch + 1) * DEV_BATCH_SIZE
                    res = model(forward_ids=dev_encoded["input_ids"][sb: eb].to(DEVICE),
                                backward_ids=dev_rev_encoded["input_ids"][sb: eb].to(DEVICE),
                                forward_kwargs={k: dev_encoded[k][sb: eb].to(DEVICE) for k in ["attention_mask"]},
                                backward_kwargs={k: dev_rev_encoded[k][sb: eb].to(DEVICE) for k in ["attention_mask"]},
                                label=dev_labels[sb: eb].to(DEVICE))

                    dev_loss += float(res["loss"])
                    dev_denom += 1

                    preds.append(torch.argmax(res["logits"], dim=-1).cpu())

                preds = torch.cat(preds)
                dev_f1 = f1_score(y_true=dev_labels.numpy(), y_pred=preds.numpy())

                logging.info(f"Dev loss: {dev_loss / max(1, dev_denom): .4f}")
                logging.info(f"Dev F1 score: {dev_f1: .4f}")

                if dev_f1 > best_f1:
                    logging.info(f"Saving new best!")
                    torch.save(model.state_dict(), os.path.join(args.experiment_dir, "weights.th"))

                    best_f1 = dev_f1
                    no_increase = 0
                else:
                    no_increase += 1

                if no_increase == args.early_stopping_rounds:
                    break

            if no_increase == args.early_stopping_rounds:
                break

    logging.info("Evaluating model on test set!")
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, "weights.th"),
                                     map_location=DEVICE))
    num_test_batches = (len(test_encoded["input_ids"]) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE
    with torch.no_grad():
        model.eval()
        test_preds = []
        test_debug = {
            "seq1": test_seq1,
            "seq2": test_seq2,
            "gold_label": test_labels,
            "predicted_label": [],
            "l2r_internal": [],
            "r2l_internal": [],
        }

        for idx_test_batch in tqdm(range(num_test_batches), total=num_dev_batches):
            sb, eb = idx_test_batch * DEV_BATCH_SIZE, (idx_test_batch + 1) * DEV_BATCH_SIZE
            res = model(forward_ids=test_encoded["input_ids"][sb: eb].to(DEVICE),
                        backward_ids=test_rev_encoded["input_ids"][sb: eb].to(DEVICE),
                        forward_kwargs={k: test_encoded[k][sb: eb].to(DEVICE) for k in ["attention_mask"]},
                        backward_kwargs={k: test_rev_encoded[k][sb: eb].to(DEVICE) for k in ["attention_mask"]})

            bert_res = model.model(test_encoded["input_ids"][sb: eb].to(DEVICE),
                                   **{k: test_encoded[k][sb: eb].to(DEVICE) for k in ["attention_mask"]})
            l2r_probas = torch.softmax(bert_res["logits"], dim=-1).cpu()
            test_debug["l2r_internal"].append(l2r_probas)

            bert_res = model.model(test_rev_encoded["input_ids"][sb: eb].to(DEVICE),
                                   **{k: test_rev_encoded[k][sb: eb].to(DEVICE) for k in ["attention_mask"]})
            r2l_probas = torch.softmax(bert_res["logits"], dim=-1)
            test_debug["r2l_internal"].append(r2l_probas)

            test_preds.append(torch.argmax(res["logits"], dim=-1).cpu())

        test_preds = torch.cat(test_preds)
        test_f1 = f1_score(y_true=test_labels.numpy(), y_pred=test_preds.numpy())
        logging.info(f"Test F1 score: {dev_f1: .4f}")

        test_debug["predicted_label"] = test_preds.tolist()
        test_debug["l2r_internal"] = torch.cat(test_debug["l2r_internal"]).tolist()
        test_debug["r2l_internal"] = torch.cat(test_debug["r2l_internal"]).tolist()

        pd.DataFrame(test_debug).to_csv(os.path.join(args.experiment_dir, "test_debug.tsv"), sep="\t", index=False)
