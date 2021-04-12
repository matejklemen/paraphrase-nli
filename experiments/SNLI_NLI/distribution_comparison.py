import json
import logging
import os
import sys
from argparse import ArgumentParser

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, \
    BertForSequenceClassification, RobertaForSequenceClassification, XLMRobertaForSequenceClassification

from src.data.nli import SNLITransformersDataset

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="distribution_debug")
parser.add_argument("--pretrained_name_or_path", type=str,
                    default="/home/matej/Documents/embeddia/paraphrasing/nli2paraphrases/models/SNLI_NLI/snli-roberta-base-combined-2e-5-maxlen42")
parser.add_argument("--model_type", type=str, default="roberta",
                    choices=["bert", "roberta", "xlm-roberta"])

parser.add_argument("--max_seq_len", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument("--use_cpu", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.use_cpu and not torch.cuda.is_available():
        raise ValueError("Device set to 'cuda' but no CUDA device is available!")
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")

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
        model_cls = BertForSequenceClassification

        def extract_embeddings(model_obj: BertForSequenceClassification, **data_dict):
            outputs = model_obj.bert(input_ids=data_dict["input_ids"],
                                     token_type_ids=data_dict["token_type_ids"],
                                     attention_mask=data_dict["attention_mask"],
                                     return_dict=True)
            return outputs["pooler_output"]

    elif args.model_type == "roberta":
        tokenizer_cls = RobertaTokenizerFast
        model_cls = RobertaForSequenceClassification

        def extract_embeddings(model_obj: RobertaForSequenceClassification, **data_dict):
            outputs = model_obj.roberta(input_ids=data_dict["input_ids"],
                                        attention_mask=data_dict["attention_mask"],
                                        return_dict=True)
            pooled_output = outputs["last_hidden_state"][:, 0, :]
            return pooled_output

    elif args.model_type == "xlm-roberta":
        tokenizer_cls = XLMRobertaTokenizerFast
        model_cls = XLMRobertaForSequenceClassification

        def extract_embeddings(model_obj: XLMRobertaForSequenceClassification, **data_dict):
            outputs = model_obj.roberta(input_ids=data_dict["input_ids"],
                                        attention_mask=data_dict["attention_mask"],
                                        return_dict=True)
            pooled_output = outputs["last_hidden_state"][:, 0, :]
            return pooled_output

    else:
        raise NotImplementedError(f"Model_type '{args.model_type}' is not supported")

    model = model_cls.from_pretrained(args.pretrained_name_or_path, return_dict=True).to(DEVICE)
    model.eval()
    tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)

    all_data = {}
    for dataset_name in ["test"]:
        dataset = SNLITransformersDataset(dataset_name, tokenizer=tokenizer,
                                          max_length=args.max_seq_len, return_tensors="pt")

        embeddings, distr_labels = [], []
        for curr_sequences, curr_label in [(zip(dataset.str_premise, dataset.str_hypothesis), 0),
                                           (zip(dataset.str_hypothesis, dataset.str_premise), 1)]:
            sequences = list(curr_sequences)
            num_batches = (len(sequences) + args.batch_size - 1) // args.batch_size

            encoded = tokenizer.batch_encode_plus(sequences, max_length=args.max_seq_len, padding="max_length",
                                                  truncation="longest_first", return_tensors="pt")

            with torch.no_grad():
                for idx_batch in tqdm(range(num_batches), total=num_batches):
                    s_b, e_b = idx_batch * args.batch_size, (idx_batch + 1) * args.batch_size

                    seq_repr = extract_embeddings(model,
                                                  **{k: v[s_b: e_b].to(DEVICE) for k, v in encoded.items()}).cpu()

                    embeddings.append(seq_repr)
                    distr_labels.extend([curr_label] * seq_repr.shape[0])

        embeddings = torch.cat(embeddings)
        distr_labels = torch.tensor(distr_labels, dtype=torch.long)

        shuf_inds = torch.randperm(embeddings.shape[0])
        embeddings = embeddings[shuf_inds]
        distr_labels = distr_labels[shuf_inds]

        all_data[dataset_name] = {"X": embeddings.numpy(), "y": distr_labels.numpy()}

    num_examples = len(all_data["test"]["X"])

    # Split the embedded test examples into a training, validation and test set
    indices = torch.randperm(num_examples).numpy()
    train_inds = indices[:int(0.8 * num_examples)]
    dev_inds = indices[int(0.8 * num_examples): int(0.9 * num_examples)]
    test_inds = indices[int(0.9 * num_examples):]
    train_X, train_y = all_data["test"]["X"][train_inds], all_data["test"]["y"][train_inds]
    dev_X, dev_y = all_data["test"]["X"][dev_inds], all_data["test"]["y"][dev_inds]
    test_X, test_y = all_data["test"]["X"][test_inds], all_data["test"]["y"][test_inds]
    logging.info(f"{len(train_inds)} training examples, {len(dev_inds)} dev examples, {len(test_inds)} test examples")

    for n_estimators in [50, 100, 500, 1000]:
        logging.info(f"Fitting random forest with n_estimators={n_estimators}")
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        model.fit(train_X, train_y)

        dev_preds = model.predict(dev_X)
        logging.info(f"\tDev accuracy: {accuracy_score(y_true=dev_y, y_pred=dev_preds): .4f}")

        test_preds = model.predict(test_X)
        logging.info(f"\tTest accuracy: {accuracy_score(y_true=test_y, y_pred=test_preds): .4f}")
        logging.info(f"\tTest majority: {test_y.sum() / len(test_y): .4f}")
