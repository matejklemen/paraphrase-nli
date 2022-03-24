import csv
import json
import logging
import os
import sys
from argparse import ArgumentParser
from time import time

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.data.paraphrase import MSCOCOTransformersDataset, TapacoTransformersDataset
from src.models.nli_trainer import TransformersNLITrainer
from src.visualization.visualize import multicolumn_visualization

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug_filtering")
parser.add_argument("--pretrained_name_or_path", type=str,
                    default="/home/matej/Documents/paraphrase-nli/models/TaPaCo_para/en-tapaco-roberta-base-2e-5-maxlength35")
parser.add_argument("--model_type", type=str, default="roberta",
                    choices=["bert", "roberta", "xlm-roberta"])
parser.add_argument("--max_seq_len", type=int, default=35)
parser.add_argument("--batch_size", type=int, default=16,
                    help="Evaluation batch size. Note that this can generally be set much higher than in training mode")

parser.add_argument("--reverse_order", action="store_true")

parser.add_argument("--train_path", type=str, help="Path to the training set of TaPaCo (tsv)", default=None)
parser.add_argument("--dev_path", type=str, help="Path to the validation set of TaPaCo (tsv)", default=None)
parser.add_argument("--test_path", type=str, help="Path to the test set of TaPaCo (tsv)", default=None)

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
        valid_preds = torch.gt(mean_proba[:, test_set.label2idx["not_paraphrase"]], thresh)

        preds[valid_preds] = test_set.label2idx["not_paraphrase"]
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
    tokenizer_cls = AutoTokenizer
    model = TransformersNLITrainer.from_pretrained(args.pretrained_name_or_path,
                                                   device=("cpu" if args.use_cpu else "cuda"),
                                                   batch_size=args.batch_size)
    tokenizer = tokenizer_cls.from_pretrained(args.pretrained_name_or_path)

    model_metrics = {}

    all_false_paras = {
        "sequence1": [],
        "sequence2": []
    }

    t1 = time()
    for dataset_name, data_path in [("train", args.train_path), ("dev", args.dev_path), ("test", args.test_path)]:
        if data_path is None:
            logging.info(f"Path for dataset '{dataset_name}' not provided, skipping...")
            continue

        dataset = TapacoTransformersDataset(data_path, tokenizer=tokenizer,
                                            max_length=args.max_seq_len, return_tensors="pt",
                                            reverse_order=args.reverse_order)

        logging.info(f"{dataset_name}: loaded {len(dataset)} examples")
        l2r_preds = {
            "seq1": dataset.seq1,
            "seq2": dataset.seq2
        }
        mean_probas = torch.zeros(len(dataset), dtype=torch.float32)
        sd_probas = torch.zeros_like(mean_probas)
        l2r_labels = torch.zeros(len(dataset), dtype=torch.long)

        if args.l2r_strategy == "ground_truth":
            assert hasattr(dataset, "labels")
            para_mask = torch.eq(dataset.labels, dataset.label2idx["paraphrase"])
            mean_probas[para_mask] = 1.0
            l2r_labels = dataset.labels
        else:
            res = get_predictions(model, dataset,
                                  pred_strategy=args.l2r_strategy,
                                  thresh=args.l2r_thresh,
                                  num_mcd_rounds=args.l2r_mcd_rounds)

            para_mask = torch.eq(res["pred_label"], dataset.label2idx["paraphrase"])
            l2r_labels = res["pred_label"]
            mean_probas = res["mean_proba"][:, dataset.label2idx["paraphrase"]]
            sd_probas = res["sd_proba"][:, dataset.label2idx["paraphrase"]]

        l2r_preds["label"] = list(map(lambda i: dataset.idx2label.get(i, "other"), l2r_labels.tolist()))
        l2r_preds["mean_proba"] = mean_probas.tolist()
        l2r_preds["sd_proba"] = sd_probas.tolist()

        logging.info(f"Writing left-to-right prediction information to file ({len(dataset)} rows)")
        pd.DataFrame(l2r_preds).to_csv(os.path.join(l2r_dir, f"{dataset_name}_preds.csv"),
                                       sep=",", index=False, quoting=csv.QUOTE_ALL)

        if "train" not in dataset_name:
            multicolumn_visualization(
                column_names=["Input", "Ground truth", f"{args.l2r_strategy} (l2r)"],
                column_values=[dataset.seq1, dataset.seq2, l2r_preds["label"]],
                column_metric_data=[None, None, {"mean(P(y=para))": l2r_preds["mean_proba"], "sd(P(y=para))": l2r_preds["sd_proba"], "y": l2r_preds["label"]}],
                global_metric_data=[None, None, {"num_para": int(torch.sum(para_mask))}],
                sort_by_system=(2, 0),  # sort by mean_proba
                path=os.path.join(l2r_dir, f"{dataset_name}_visualization.html")
            )
        else:
            logging.info(f"Skipping visualization for dataset '{dataset_name}' as will likely grow too big")

        para_inds = torch.flatten(torch.nonzero(para_mask, as_tuple=False)).tolist()
        logging.info(f"Left-to-right: {len(para_inds)} paraphrases found!")

        # Reverse the order
        new_seq1 = [dataset.seq2[_i] for _i in para_inds]
        new_seq2 = [dataset.seq1[_i] for _i in para_inds]

        encoded = tokenizer.batch_encode_plus(list(zip(new_seq1, new_seq2)),
                                              max_length=args.max_seq_len, padding="max_length",
                                              truncation="longest_first", return_tensors="pt")
        for k, v in encoded.items():
            setattr(dataset, k, v)

        dataset.seq1 = new_seq1
        dataset.seq2 = new_seq2
        dataset.num_examples = len(dataset.seq1)

        # Labels are from left-to-right annotations (i.e. invalid for reverse)
        delattr(dataset, "labels")
        dataset.valid_attrs.remove("labels")

        r2l_preds = {
            "seq1": dataset.seq1,
            "seq2": dataset.seq2
        }

        logging.info(f"Predicting probas")
        reverse_res = get_predictions(model, dataset,
                                      pred_strategy=args.r2l_strategy,
                                      thresh=args.r2l_thresh,
                                      num_mcd_rounds=args.r2l_mcd_rounds)

        # Note that for right-to-left we are interested in P(y=not_paraphrase) (filtering!)
        r2l_preds["label"] = list(map(lambda i: dataset.idx2label.get(i, "other"), reverse_res["pred_label"].tolist()))
        r2l_preds["mean_proba"] = reverse_res["mean_proba"][:, dataset.label2idx["not_paraphrase"]].tolist()
        r2l_preds["sd_proba"] = reverse_res["sd_proba"][:, dataset.label2idx["not_paraphrase"]].tolist()

        rev_nonpara_mask = torch.eq(reverse_res["pred_label"], dataset.label2idx["not_paraphrase"])
        rev_nonpara_inds = torch.flatten(torch.nonzero(rev_nonpara_mask, as_tuple=False)).tolist()

        logging.info(f"Writing right-to-left prediction information to file ({len(dataset)} rows)")
        pd.DataFrame(r2l_preds).to_csv(os.path.join(r2l_dir, f"{dataset_name}_preds.csv"),
                                       sep=",", index=False, quoting=csv.QUOTE_ALL)

        if "train" not in dataset_name:
            multicolumn_visualization(
                column_names=["Input", "Ground truth", f"{args.r2l_strategy} (r2l)"],
                column_values=[dataset.seq1, dataset.seq2, r2l_preds["label"]],
                column_metric_data=[None, None,
                                    {"mean(P(y=not_para))": r2l_preds["mean_proba"], "sd(P(y=not_para))": r2l_preds["sd_proba"],
                                     "y": r2l_preds["label"]}],
                global_metric_data=[None, None, {"num_nonpara": int(torch.sum(rev_nonpara_mask))}],
                sort_by_system=(2, 0),  # sort by mean_proba
                path=os.path.join(r2l_dir, f"{dataset_name}_visualization.html")
            )
        else:
            logging.info(f"Skipping visualization for dataset '{dataset_name}' as will likely grow too big")

        logging.info(f"{len(rev_nonpara_inds)} non-paraphrases found!")
        false_paras = {"sequence1": [], "sequence2": []}
        for _i in rev_nonpara_inds:
            # Write them in the order in which they first appeared in (in para dataset)
            false_paras["sequence1"].append(dataset.seq2[_i])
            false_paras["sequence2"].append(dataset.seq1[_i])

        logging.info(f"Writing false paraphrases for dataset '{dataset_name}' ({len(false_paras['sequence1'])} examples)")
        model_metrics[f"false_paraphrases_{dataset_name}"] = len(false_paras['sequence1'])
        pd.DataFrame(false_paras).to_csv(os.path.join(args.experiment_dir, f"{dataset_name}_false_paraphrases.csv"),
                                         sep=",", index=False, quoting=csv.QUOTE_ALL)

        all_false_paras["sequence1"].extend(false_paras["sequence1"])
        all_false_paras["sequence2"].extend(false_paras["sequence2"])

    t2 = time()
    logging.info(f"Filtering took {t2 - t1: .4f}s")
    model_metrics["time_taken"] = round(t2 - t1, 4)

    with open(os.path.join(args.experiment_dir, "metrics.json"), "w") as f_metrics:
        logging.info(model_metrics)
        json.dump(model_metrics, fp=f_metrics, indent=4)

    logging.info(f"Writing combined false paraphrases ({len(all_false_paras['sequence1'])} examples)")
    pd.DataFrame(all_false_paras).to_csv(os.path.join(args.experiment_dir, f"all_false_paraphrases.csv"),
                                         sep=",", index=False, quoting=csv.QUOTE_ALL)
