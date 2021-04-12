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
    BertModel, RobertaModel, XLMRobertaModel

from src.data.nli import SNLITransformersDataset

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="distribution_debug")
parser.add_argument("--pretrained_name_or_path", type=str,
                    default="/home/matej/Documents/embeddia/paraphrasing/nli2paraphrases/models/SNLI_NLI/snli-bert-base-cased-combined-2e-5-maxlen51")
parser.add_argument("--model_type", type=str, default="bert",
                    choices=["bert", "roberta", "xlm-roberta"])

parser.add_argument("--max_seq_len", type=int, default=41)
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
        embedder_cls = BertModel
    elif args.model_type == "roberta":
        tokenizer_cls = RobertaTokenizerFast
        embedder_cls = RobertaModel
    elif args.model_type == "xlm-roberta":
        tokenizer_cls = XLMRobertaTokenizerFast
        embedder_cls = XLMRobertaModel
    else:
        raise NotImplementedError("Model_type '{args.model_type}' is not supported")

    model = embedder_cls.from_pretrained(args.pretrained_name_or_path, return_dict=True).to(DEVICE)
    tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)

    all_data = {}
    for dataset_name in ["train", "validation", "test"]:
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

                    res = model(**{k: v[s_b: e_b].to(DEVICE) for k, v in encoded.items()})
                    pooler_output = res["pooler_output"].cpu()

                    embeddings.append(pooler_output)
                    distr_labels.extend([curr_label] * pooler_output.shape[0])

        embeddings = torch.cat(embeddings)
        distr_labels = torch.tensor(distr_labels, dtype=torch.long)

        shuf_inds = torch.randperm(embeddings.shape[0])
        embeddings = embeddings[shuf_inds]
        distr_labels = distr_labels[shuf_inds]

        all_data[dataset_name] = {"X": embeddings.numpy(), "y": distr_labels.numpy()}

    for n_estimators in [50, 100, 500, 1000]:
        logging.info(f"Fitting random forest with n_estimators={n_estimators}")
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        model.fit(all_data["train"]["X"], all_data["train"]["y"])

        val_preds = model.predict(all_data["validation"]["X"])
        logging.info(f"\tDev accuracy: {accuracy_score(y_true=all_data['validation']['y'], y_pred=val_preds)}")

        test_preds = model.predict(all_data["test"]["X"])
        logging.info(f"\tTest accuracy: {accuracy_score(y_true=all_data['test']['y'], y_pred=test_preds)}")
