import csv
import json
import logging
import os
import sys
from argparse import ArgumentParser
from time import time

import pandas as pd
import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast

from src.data.nli import OCNLITransformersDataset
from src.models.nli_trainer import TransformersNLITrainer
from src.visualization.visualize import multicolumn_visualization

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug_extraction")
parser.add_argument("--pretrained_name_or_path", type=str,
                    default="/home/matej/Documents/paraphrase-nli/models/OCNLI/ocnli-chinese-roberta-wwm-ext-2e-5-maxlength62")
parser.add_argument("--model_type", type=str, default="bert",
                    choices=["bert", "roberta", "xlm-roberta"])
parser.add_argument("--max_seq_len", type=int, default=62)
parser.add_argument("--batch_size", type=int, default=16,
                    help="Evaluation batch size. Note that this can generally be set much higher than in training mode")

parser.add_argument("--train_path", type=str, help="Path to the training set of OCNLI (json)",
                    default="/home/matej/Documents/data/ocnli/train.50k.json")
parser.add_argument("--dev_path", type=str, help="Path to the validation set of OCNLI (json)",
                    default="/home/matej/Documents/data/ocnli/dev.json")
parser.add_argument("--binary_task", action="store_true",
                    help="If set, convert the NLI task into a RTE task, i.e. predicting whether y == entailment or not")

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
        preds = -1 * torch.ones(mean_proba.shape[0], dtype=torch.long)
        valid_preds = torch.gt(mean_proba[:, test_set.label2idx["entailment"]], thresh)

        preds[valid_preds] = test_set.label2idx["entailment"]
    else:
        raise NotImplementedError()

    model.use_mcd = prev_value

    return {
        "pred_label": preds,  # -1 or [0, model.num_labels)
        "mean_proba": mean_proba,
        "sd_proba": sd_proba
    }


if __name__ == "__main__":
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
    elif args.model_type == "roberta":
        tokenizer_cls = RobertaTokenizerFast
    elif args.model_type == "xlm-roberta":
        tokenizer_cls = XLMRobertaTokenizerFast
    else:
        raise NotImplementedError(f"Model_type '{args.model_type}' is not supported")

    model = TransformersNLITrainer.from_pretrained(args.pretrained_name_or_path,
                                                   device=("cpu" if args.use_cpu else "cuda"),
                                                   batch_size=args.batch_size)
    tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)

    if args.binary_task:
        assert model.num_labels == 2

    model_metrics = {}

    all_paras = {
        "sequence1": [],
        "sequence2": []
    }

    t1 = time()
    for dataset_name, dataset_path in [("train", args.train_path), ("validation", args.dev_path)]:
        dataset = OCNLITransformersDataset(dataset_path, tokenizer=tokenizer,
                                           max_length=args.max_seq_len, return_tensors="pt",
                                           binarize=args.binary_task)
        logging.info(f"{dataset_name}: loaded {len(dataset)} examples")
        l2r_preds = {
            "premise": dataset.str_premise,
            "hypothesis": dataset.str_hypothesis
        }
        mean_probas = torch.zeros(len(dataset), dtype=torch.float32)
        sd_probas = torch.zeros_like(mean_probas)
        l2r_labels = torch.zeros(len(dataset), dtype=torch.long)

        if args.l2r_strategy == "ground_truth":
            assert hasattr(dataset, "labels")
            ent_mask = dataset.labels == dataset.label2idx["entailment"]
            mean_probas[ent_mask] = 1.0
            l2r_labels = dataset.labels
        else:
            res = get_predictions(model, dataset,
                                  pred_strategy=args.l2r_strategy,
                                  thresh=args.l2r_thresh,
                                  num_mcd_rounds=args.l2r_mcd_rounds)

            ent_mask = (res["pred_label"] == dataset.label2idx["entailment"])
            l2r_labels = res["pred_label"]
            mean_probas = res["mean_proba"][:, dataset.label2idx["entailment"]]
            sd_probas = res["sd_proba"][:, dataset.label2idx["entailment"]]

        l2r_preds["label"] = list(map(lambda i: dataset.label_names[i], l2r_labels.tolist()))
        l2r_preds["mean_proba"] = mean_probas.tolist()
        l2r_preds["sd_proba"] = sd_probas.tolist()

        logging.info(f"Writing left-to-right prediction information to file ({len(dataset)} rows)")
        pd.DataFrame(l2r_preds).to_csv(os.path.join(l2r_dir, f"{dataset_name}_preds.csv"),
                                       sep=",", index=False, quoting=csv.QUOTE_ALL)
        if "train" not in dataset_name:
            multicolumn_visualization(
                column_names=["Input", "Ground truth", f"{args.l2r_strategy} (l2r)"],
                column_values=[dataset.str_premise, dataset.str_hypothesis, l2r_preds["label"]],
                column_metric_data=[None, None, {"mean(P(y=ent))": l2r_preds["mean_proba"], "sd(P(y=ent))": l2r_preds["sd_proba"], "y": l2r_preds["label"]}],
                global_metric_data=[None, None, {"num_entailment": int(torch.sum(ent_mask))}],
                sort_by_system=(2, 0),  # sort by mean_proba
                path=os.path.join(l2r_dir, f"{dataset_name}_visualization.html")
            )
        else:
            logging.info(f"Skipping visualization for dataset '{dataset_name}' as will likely grow too big")

        ent_inds = torch.flatten(torch.nonzero(ent_mask, as_tuple=False)).tolist()

        # Reverse the order
        new_prem = [dataset.str_hypothesis[_i] for _i in ent_inds]
        new_hyp = [dataset.str_premise[_i] for _i in ent_inds]

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
        r2l_preds["label"] = list(map(lambda i: dataset.label_names[i], reverse_res["pred_label"].tolist()))
        r2l_preds["mean_proba"] = reverse_res["mean_proba"][:, dataset.label2idx["entailment"]].tolist()
        r2l_preds["sd_proba"] = reverse_res["sd_proba"][:, dataset.label2idx["entailment"]].tolist()

        rev_ent_mask = reverse_res["pred_label"] == dataset.label2idx["entailment"]  # type: torch.tensor
        rev_ent_inds = torch.flatten(torch.nonzero(rev_ent_mask, as_tuple=False)).tolist()

        logging.info(f"Writing right-to-left prediction information to file ({len(dataset)} rows)")
        pd.DataFrame(r2l_preds).to_csv(os.path.join(r2l_dir, f"{dataset_name}_preds.csv"),
                                       sep=",", index=False, quoting=csv.QUOTE_ALL)

        if "train" not in dataset_name:
            multicolumn_visualization(
                column_names=["Input", "Ground truth", f"{args.r2l_strategy} (r2l)"],
                column_values=[dataset.str_premise, dataset.str_hypothesis, r2l_preds["label"]],
                column_metric_data=[None, None,
                                    {"mean(P(y=ent))": r2l_preds["mean_proba"], "sd(P(y=ent))": r2l_preds["sd_proba"],
                                     "y": r2l_preds["label"]}],
                global_metric_data=[None, None, {"num_entailment": int(torch.sum(rev_ent_mask))}],
                sort_by_system=(2, 0),  # sort by mean_proba
                path=os.path.join(r2l_dir, f"{dataset_name}_visualization.html")
            )
        else:
            logging.info(f"Skipping visualization for dataset '{dataset_name}' as will likely grow too big")

        logging.info(f"{len(rev_ent_inds)} paraphrases found!")
        paras = {"sequence1": [], "sequence2": []}
        for _i in rev_ent_inds:
            # Write them in the order in which they first appeared in (in NLI dataset)
            paras["sequence1"].append(dataset.str_hypothesis[_i])
            paras["sequence2"].append(dataset.str_premise[_i])

        logging.info(f"Writing paraphrases for dataset '{dataset_name}' ({len(paras['sequence1'])} examples)")
        model_metrics[f"paraphrases_{dataset_name}"] = len(paras['sequence1'])
        pd.DataFrame(paras).to_csv(os.path.join(args.experiment_dir, f"{dataset_name}_paraphrases.csv"),
                                   sep=",", index=False, quoting=csv.QUOTE_ALL)

        all_paras["sequence1"].extend(paras["sequence1"])
        all_paras["sequence2"].extend(paras["sequence2"])

    t2 = time()
    logging.info(f"Extraction took {t2 - t1: .4f}s")
    model_metrics["time_taken"] = round(t2 - t1, 4)

    with open(os.path.join(args.experiment_dir, "metrics.json"), "w") as f_metrics:
        logging.info(model_metrics)
        json.dump(model_metrics, fp=f_metrics, indent=4)

    model_metrics["paraphrases_total"] = len(all_paras['sequence1'])
    logging.info(f"Writing combined paraphrases ({len(all_paras['sequence1'])} examples)")
    pd.DataFrame(all_paras).to_csv(os.path.join(args.experiment_dir, f"all_paraphrases.csv"),
                                   sep=",", index=False, quoting=csv.QUOTE_ALL)
