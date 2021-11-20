import csv
import json
import logging
import os
import sys
from argparse import ArgumentParser
from time import time

import pandas as pd
import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, AutoTokenizer, \
    CamembertTokenizerFast

from src.data.nli import XNLITransformersDataset
from src.models.nli_trainer import TransformersNLITrainer
from src.visualization.visualize import multicolumn_visualization

parser = ArgumentParser()
parser.add_argument("--lang", type=str, default="de")
parser.add_argument("--experiment_dir", type=str, default="debug_extraction")
parser.add_argument("--pretrained_name_or_path", type=str)
parser.add_argument("--model_type", type=str, default="bert",
                    choices=["bert", "roberta", "camembert", "xlm-roberta", "phobert"])
parser.add_argument("--max_seq_len", type=int, default=98)
parser.add_argument("--batch_size", type=int, default=16,
                    help="Evaluation batch size. Note that this can generally be set much higher than in training mode")

parser.add_argument("--binary_task", action="store_true",
                    help="If set, convert the NLI task into a RTE task, i.e. predicting whether y == entailment or not")

parser.add_argument("--only_train_set", action="store_true",
                    help="If set, extract paraphrases from training set, otherwise extract them from validation and "
                         "test set")

parser.add_argument("--custom_dev_path", type=str, default=None,
                    help="If set to a path, will load XNLI dev set from this path instead of from 'datasets' library")
parser.add_argument("--custom_test_path", type=str, default=None,
                    help="If set to a path, will load XNLI test set from this path instead of from 'datasets' library")

parser.add_argument("--l2r_strategy", choices=["ground_truth", "argmax", "thresh"], default="ground_truth")
parser.add_argument("--r2l_strategy", choices=["argmax", "thresh"], default="argmax")
parser.add_argument("--l2r_thresh", type=float, default=None, help="Optional (used if l2r_strategy is 'thresh')")
parser.add_argument("--r2l_thresh", type=float, default=None, help="Optional (used if r2l_strategy is 'thresh')")

parser.add_argument("--l2r_mcd_rounds", type=int, default=0)
parser.add_argument("--r2l_mcd_rounds", type=int, default=0)

parser.add_argument("--use_cpu", action="store_true")


def get_predictions(model, test_set, pred_strategy="argmax", num_mcd_rounds=0, thresh=None):
    prev_value = model.use_mcd
    model.use_mcd = num_mcd_rounds > 0
    num_reps = 1 if num_mcd_rounds == 0 else int(num_mcd_rounds)
    probas = []

    for idx_rep in range(num_reps):
        logging.info(f"Running evaluation iteration#{idx_rep}")
        test_res = model.evaluate(test_set)  # [num_examples, num_classes]
        probas.append(test_res["pred_proba"])

    probas = torch.stack(probas)
    mean_proba = torch.mean(probas, dim=0)
    sd_proba = torch.zeros_like(mean_proba) if num_reps == 1 else torch.std(probas, dim=0)

    if pred_strategy == "argmax":
        preds = torch.argmax(mean_proba, dim=-1)
    elif pred_strategy == "thresh":
        assert thresh is not None
        highest_proba_class = torch.argmax(mean_proba, dim=-1)
        preds = -1 * torch.ones(mean_proba.shape[0], dtype=torch.long)
        valid_preds = torch.gt(mean_proba[torch.arange(mean_proba.shape[0]), highest_proba_class], thresh)

        preds[valid_preds] = highest_proba_class[valid_preds]
    else:
        raise NotImplementedError()

    model.use_mcd = prev_value

    return {
        "pred_label": preds,  # -1 or [0, model.num_labels)
        "mean_proba": mean_proba,
        "sd_proba": sd_proba
    }


if __name__ == "__main__":
    ALL_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    args = parser.parse_args()
    l2r_dir = os.path.join(args.experiment_dir, "l2r")
    r2l_dir = os.path.join(args.experiment_dir, "r2l")
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
        os.makedirs(l2r_dir)
        os.makedirs(r2l_dir)

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

    model = TransformersNLITrainer.from_pretrained(args.pretrained_name_or_path,
                                                   device=("cpu" if args.use_cpu else "cuda"),
                                                   batch_size=args.batch_size)
    tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)
    tokenizer.save_pretrained(args.experiment_dir)

    if args.binary_task:
        assert model.num_labels == 2

    if args.lang == "all_languages":
        if args.custom_dev_path is not None and args.custom_test_path is not None:
            df_dev = pd.read_csv(args.custom_dev_path, sep="\t")
            df_test = pd.read_csv(args.custom_test_path, sep="\t")
            datasets_to_process = []

            # A bit of a hack hardcoding this, so we assert that binary task is not being done
            assert not args.binary_task
            label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

            for curr_dev_lang, curr_dev_group in df_dev.groupby("language"):
                datasets_to_process.append({
                    "lang": curr_dev_lang,
                    "split": "validation",
                    "override_data": {
                        "sentence1": curr_dev_group["sentence1"].tolist(),
                        "sentence2": curr_dev_group["sentence2"].tolist(),
                        "label": curr_dev_group["gold_label"].apply(lambda label_str: label_map[label_str]).tolist()
                    }
                })

            for curr_test_lang, curr_test_group in df_test.groupby("language"):
                datasets_to_process.append({
                    "lang": curr_test_lang,
                    "split": "test",
                    "override_data": {
                        "sentence1": curr_test_group["sentence1"].tolist(),
                        "sentence2": curr_test_group["sentence2"].tolist(),
                        "label": curr_test_group["gold_label"].apply(lambda label_str: label_map[label_str]).tolist()
                    }
                })
        else:
            datasets_to_process = [{
                "lang": curr_lang,
                "split": curr_split
            } for curr_lang in ALL_LANGS for curr_split in ["validation", "test"]]
    else:
        if args.only_train_set:
            logging.info("Note: only processing train set!")
            datasets_to_process = [{
                "lang": args.lang,
                "split": "train",
                "preloaded_dataset": XNLITransformersDataset(args.lang, "train", tokenizer=tokenizer,
                                                             max_length=args.max_seq_len, return_tensors="pt",
                                                             binarize=args.binary_task)
            }]
        else:
            logging.info("Note: only processing validation and test set!")
            datasets_to_process = [{
                "lang": args.lang,
                "split": curr_split,
                "preloaded_dataset": XNLITransformersDataset(args.lang, curr_split, tokenizer=tokenizer,
                                                             max_length=args.max_seq_len, return_tensors="pt",
                                                             binarize=args.binary_task)
            } for curr_split in ["validation", "test"]]

    model_metrics = {}

    t1 = time()
    for curr_dataset_metadata in datasets_to_process:
        lang, split = curr_dataset_metadata["lang"], curr_dataset_metadata["split"]
        dataset_name = f"{lang}_{split}"
        if "preloaded_dataset" in curr_dataset_metadata:
            dataset = curr_dataset_metadata["preloaded_dataset"]
            logging.info(f"'{dataset_name}': Using pre-loaded dataset with {len(dataset)} examples")
        else:
            dataset = XNLITransformersDataset(lang, split, tokenizer=tokenizer,
                                              max_length=args.max_seq_len, return_tensors="pt",
                                              binarize=args.binary_task)
            if "override_data" in curr_dataset_metadata:
                dataset_name = f"{lang}_{split}_custom"
                dataset.override_data(new_seq1=curr_dataset_metadata["override_data"]["sentence1"],
                                      new_seq2=curr_dataset_metadata["override_data"]["sentence2"],
                                      new_labels=curr_dataset_metadata["override_data"]["label"])
                logging.info(f"'{dataset_name}': Using overriden dataset with {len(dataset)} examples")
            else:
                logging.info(f"'{dataset_name}': Using loaded regular dataset with {len(dataset)} examples")

        l2r_preds = {
            "premise": dataset.str_premise,
            "hypothesis": dataset.str_hypothesis
        }
        mean_probas = torch.zeros(len(dataset), dtype=torch.float32)
        sd_probas = torch.zeros_like(mean_probas)
        l2r_labels = torch.zeros(len(dataset), dtype=torch.long)

        if args.l2r_strategy == "ground_truth":
            assert hasattr(dataset, "labels")
            mean_probas[:] = 1.0
            sd_probas[:] = 0.0
            l2r_labels = dataset.labels
        else:
            res = get_predictions(model, dataset,
                                  pred_strategy=args.l2r_strategy,
                                  thresh=args.l2r_thresh,
                                  num_mcd_rounds=args.l2r_mcd_rounds)

            l2r_labels = res["pred_label"]
            mean_probas = res["mean_proba"][torch.arange(len(dataset)), l2r_labels]
            sd_probas = res["sd_proba"][torch.arange(len(dataset)), l2r_labels]

        l2r_preds["label"] = list(map(lambda i: dataset.label2idx.get(i, "other"), l2r_labels.tolist()))
        l2r_preds["mean_proba"] = mean_probas.tolist()
        l2r_preds["sd_proba"] = sd_probas.tolist()

        logging.info(f"Writing left-to-right prediction information to file ({len(dataset)} rows)")
        pd.DataFrame(l2r_preds).to_csv(os.path.join(l2r_dir, f"{dataset_name}_preds.csv"),
                                       sep=",", index=False, quoting=csv.QUOTE_ALL)
        if "train" not in dataset_name:
            multicolumn_visualization(
                column_names=["Input", "Ground truth", f"{args.l2r_strategy} (l2r)"],
                column_values=[dataset.str_premise, dataset.str_hypothesis, l2r_preds["label"]],
                column_metric_data=[None, None, {"mean(P(y=y_hat))": l2r_preds["mean_proba"],
                                                 "sd(P(y=y_hat))": l2r_preds["sd_proba"], "y_hat": l2r_preds["label"]}],
                sort_by_system=(2, 0),  # sort by mean_proba
                path=os.path.join(l2r_dir, f"{dataset_name}_visualization.html")
            )
        else:
            logging.info(f"Skipping visualization for dataset '{dataset_name}' as will likely grow too big")

        # Reverse the order
        new_prem = dataset.str_hypothesis
        new_hyp = dataset.str_premise

        encoded = tokenizer.batch_encode_plus(list(zip(new_prem, new_hyp)),
                                              max_length=args.max_seq_len, padding="max_length",
                                              truncation="longest_first", return_tensors="pt")
        for k, v in encoded.items():
            setattr(dataset, k, v)

        dataset.str_premise = new_prem
        dataset.str_hypothesis = new_hyp
        dataset.num_examples = len(dataset.str_premise)

        # Labels are from left-to-right annotations (i.e. invalid for reverse)
        delattr(dataset, "labels")
        dataset.valid_attrs.remove("labels")

        r2l_preds = {
            "premise": dataset.str_premise,
            "hypothesis": dataset.str_hypothesis
        }

        logging.info(f"Predicting probas")
        reverse_res = get_predictions(model, dataset,
                                      pred_strategy=args.r2l_strategy,
                                      thresh=args.r2l_thresh,
                                      num_mcd_rounds=args.r2l_mcd_rounds)
        r2l_preds["label"] = list(map(lambda i: dataset.label2idx.get(i, "other"), reverse_res["pred_label"].tolist()))
        r2l_preds["mean_proba"] = reverse_res["mean_proba"][torch.arange(len(dataset)),
                                                            reverse_res["pred_label"]].tolist()
        r2l_preds["sd_proba"] = reverse_res["sd_proba"][torch.arange(len(dataset)),
                                                        reverse_res["pred_label"]].tolist()

        logging.info(f"Writing right-to-left prediction information to file ({len(dataset)} rows)")
        pd.DataFrame(r2l_preds).to_csv(os.path.join(r2l_dir, f"{dataset_name}_preds.csv"),
                                       sep=",", index=False, quoting=csv.QUOTE_ALL)

        if "train" not in dataset_name:
            multicolumn_visualization(
                column_names=["Input", "Ground truth", f"{args.r2l_strategy} (r2l)"],
                column_values=[dataset.str_premise, dataset.str_hypothesis, r2l_preds["label"]],
                column_metric_data=[None, None,
                                    {"mean(P(y=y_hat))": r2l_preds["mean_proba"], "sd(P(y=y_hat))": r2l_preds["sd_proba"],
                                     "y_hat": r2l_preds["label"]}],
                sort_by_system=(2, 0),  # sort by mean_proba
                path=os.path.join(r2l_dir, f"{dataset_name}_visualization.html")
            )
        else:
            logging.info(f"Skipping visualization for dataset '{dataset_name}' as will likely grow too big")

        # To obtain challenging non-paraphrases, we use (ent, neutral) pairs - these are not available in binary case,
        # so we use (ent, non-ent) pairs there
        OTHER_LABEL = "not_entailment" if args.binary_task else "neutral"
        assert dataset.label2idx.get(OTHER_LABEL, None) is not None

        paras_mask = torch.logical_and(l2r_labels == dataset.label2idx["entailment"],
                                       reverse_res["pred_label"] == dataset.label2idx["entailment"])
        nonparas_mask = torch.logical_or(torch.logical_and(l2r_labels == dataset.label2idx["entailment"],
                                                           reverse_res["pred_label"] == dataset.label2idx[OTHER_LABEL]),
                                         torch.logical_and(l2r_labels == dataset.label2idx[OTHER_LABEL],
                                                           reverse_res["pred_label"] == dataset.label2idx["entailment"]))
        para_inds = torch.flatten(torch.nonzero(paras_mask, as_tuple=False)).tolist()
        nonpara_inds = torch.flatten(torch.nonzero(nonparas_mask, as_tuple=False)).tolist()
        logging.info(f"{len(para_inds)} paraphrases and {len(nonpara_inds)} (challenging) non-paraphrases found!")

        paras = {"sequence1": [], "sequence2": [], "label": []}
        # Write pairs in the order in which they first appeared in (i.e. non-reversed)
        for _i in para_inds:
            paras["sequence1"].append(dataset.str_hypothesis[_i])
            paras["sequence2"].append(dataset.str_premise[_i])
            paras["label"].append(1)

        for _i in nonpara_inds:
            paras["sequence1"].append(dataset.str_hypothesis[_i])
            paras["sequence2"].append(dataset.str_premise[_i])
            paras["label"].append(0)

        logging.info(f"Writing paraphrase identification examples for dataset '{dataset_name}' "
                     f"({len(paras['sequence1'])} examples)")
        model_metrics[f"paraphrases_{dataset_name}"] = {
            "total": len(para_inds) + len(nonpara_inds),
            "paraphrases": len(para_inds),
            "non-paraphrases": len(nonpara_inds)
        }
        pd.DataFrame(paras).to_csv(os.path.join(args.experiment_dir, f"{dataset_name}_paraphrases.csv"),
                                   sep=",", index=False, quoting=csv.QUOTE_ALL)

    t2 = time()
    logging.info(f"Extraction took {t2 - t1: .4f}s")
    model_metrics["time_taken"] = round(t2 - t1, 4)

    with open(os.path.join(args.experiment_dir, "metrics.json"), "w") as f_metrics:
        logging.info(model_metrics)
        json.dump(model_metrics, fp=f_metrics, indent=4)
