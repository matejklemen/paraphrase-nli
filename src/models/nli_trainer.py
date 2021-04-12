import json
import logging
import os
from time import time
from typing import Optional

import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification


class TransformersNLITrainer:
    def __init__(self, model_dir, pretrained_model_name_or_path, num_labels, pred_strategy="argmax", thresh=None,
                 batch_size=24, learning_rate=6.25e-5, validate_every_n_steps=5_000, early_stopping_tol=5,
                 use_mcd: Optional[bool] = False, class_weights: Optional = None,
                 optimized_metric="accuracy", device="cuda"):
        self.model_save_path = model_dir

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validate_every_n_steps = validate_every_n_steps
        self.early_stopping_tol = early_stopping_tol
        self.num_labels = num_labels

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device set to 'cuda' but no CUDA device is available!")
        self.device_str = device
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")

        self.pred_strategy = pred_strategy
        self.thresh = thresh
        if pred_strategy == "argmax":
            def predict_fn(logits):
                probas = torch.softmax(logits, dim=-1)
                return torch.argmax(probas, dim=-1)
        elif pred_strategy == "thresh":
            assert thresh is not None and 0.0 <= thresh < 1.0

            def predict_fn(logits):
                # Examples where no label has certainty above `thresh` will be labeled -1
                final_preds = -1 * torch.ones(logits.shape[0], dtype=torch.long)
                probas = torch.softmax(logits, dim=-1)
                valid_preds = torch.any(torch.gt(probas, self.thresh), dim=-1)

                # With lower threshold, it is possible that multiple labels go above it
                final_preds[valid_preds] = torch.argmax(probas[valid_preds], dim=-1)
                return final_preds
        else:
            raise NotImplementedError(f"Prediction strategy '{pred_strategy}' not supported")

        self.optimized_metric = optimized_metric
        if self.optimized_metric == "binary_f1":
            assert self.num_labels == 2

        if class_weights is None:
            self.class_weights = torch.ones(self.num_labels, dtype=torch.float32)
        else:
            assert len(class_weights) == self.num_labels
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = self.class_weights.to(self.device)

        self.use_mcd = use_mcd
        self.predict_label = predict_fn
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path,
                                                                        num_labels=self.num_labels,
                                                                        return_dict=True).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def save_pretrained(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(os.path.join(save_dir, "trainer_config.json")):
            with open(os.path.join(save_dir, "trainer_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "model_dir": save_dir,
                    "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
                    "num_labels": self.num_labels,
                    "pred_strategy": self.pred_strategy,
                    "use_mcd": self.use_mcd,
                    "thresh": self.thresh,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "validate_every_n_steps": self.validate_every_n_steps,
                    "early_stopping_tol": self.early_stopping_tol,
                    "class_weights": self.class_weights.cpu().tolist(),
                    "optimized_metric": self.optimized_metric,
                    "device": self.device_str
                }, fp=f, indent=4)

        self.model.save_pretrained(save_dir)

    @staticmethod
    def from_pretrained(model_dir, **config_override_kwargs):
        with open(os.path.join(model_dir, "trainer_config.json"), "r", encoding="utf-8") as f:
            pretrained_config = json.load(f)

        pretrained_config["pretrained_model_name_or_path"] = model_dir

        for k in config_override_kwargs:
            logging.info(f"from_pretrained: overriding '{k}' ({k}={config_override_kwargs[k]})")
            pretrained_config[k] = config_override_kwargs[k]

        instance = TransformersNLITrainer(**pretrained_config)
        return instance

    def train(self, train_dataset):
        criterion = CrossEntropyLoss(weight=self.class_weights)

        self.model.train()
        num_batches = (len(train_dataset) + self.batch_size - 1) // self.batch_size
        train_loss = 0.0
        for curr_batch in tqdm(DataLoader(train_dataset, shuffle=False, batch_size=self.batch_size),
                               total=num_batches):
            res = self.model(**{k: v.to(self.device) for k, v in curr_batch.items()})
            loss = criterion(res["logits"].view(-1, self.num_labels), curr_batch["labels"].view(-1))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += float(loss)

        return {"train_loss": train_loss}

    @torch.no_grad()
    def evaluate(self, val_dataset):
        if self.use_mcd:
            self.model.train()
        else:
            self.model.eval()

        num_batches = (len(val_dataset) + self.batch_size - 1) // self.batch_size
        eval_loss = 0.0
        compute_loss = hasattr(val_dataset, "labels")

        results = {
            "pred_label": [],
            "pred_proba": []
        }
        for curr_batch in tqdm(DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size),
                               total=num_batches):
            res = self.model(**{k: v.to(self.device) for k, v in curr_batch.items()})
            if compute_loss:
                eval_loss += float(res["loss"])
            num_batches += 1

            probas = torch.softmax(res["logits"], dim=-1)
            preds = self.predict_label(logits=res["logits"])

            results["pred_label"].append(preds.cpu())
            results["pred_proba"].append(probas.cpu())

        results["pred_label"] = torch.cat(results["pred_label"])
        results["pred_proba"] = torch.cat(results["pred_proba"])
        if compute_loss:
            results["eval_loss"] = eval_loss / max(num_batches, 1)
        return results

    def run(self, train_dataset, val_dataset, num_epochs):
        best_metric, no_increase = float("inf") if self.optimized_metric == "loss" else -float("inf"), 0
        stop_early = False

        train_start = time()
        for idx_epoch in range(num_epochs):
            logging.info(f"Epoch {1+idx_epoch}/{num_epochs}")
            shuffled_indices = torch.randperm(len(train_dataset))
            train_loss, nb = 0.0, 0

            num_minisets = (len(train_dataset) + self.validate_every_n_steps - 1) // self.validate_every_n_steps
            for idx_miniset in range(num_minisets):
                logging.info(f"Miniset {1+idx_miniset}/{num_minisets}")
                curr_subset = Subset(train_dataset, shuffled_indices[idx_miniset * self.validate_every_n_steps:
                                                                     (idx_miniset + 1) * self.validate_every_n_steps])
                num_subset_batches = (len(curr_subset) + self.batch_size - 1) // self.batch_size
                train_res = self.train(curr_subset)
                train_loss += train_res["train_loss"]
                nb += num_subset_batches
                logging.info(f"Training loss = {train_loss / nb: .4f}")

                if val_dataset is None or len(curr_subset) < self.validate_every_n_steps // 2:
                    logging.info(f"Skipping validation after training on a small training subset "
                                 f"({len(curr_subset)} < {self.validate_every_n_steps // 2} examples)")
                    continue

                val_res = self.evaluate(val_dataset)
                val_loss = val_res["eval_loss"]
                logging.info(f"Validation loss = {val_loss: .4f}")

                if self.optimized_metric == "loss":
                    is_better = val_loss < best_metric
                    val_metric = val_loss
                elif self.optimized_metric == "binary_f1":
                    val_acc = float(torch.sum(torch.eq(val_res["pred_label"], val_dataset.labels))) / len(val_dataset)
                    logging.info(f"(Not being optimized) Validation accuracy: {val_acc: .4f}")

                    val_f1 = f1_score(y_true=val_dataset.labels.cpu().numpy(),
                                      y_pred=val_res["pred_label"].cpu().numpy())
                    logging.info(f"Validation binary F1: {val_f1: .4f}")
                    is_better = val_f1 > best_metric
                    val_metric = val_f1
                else:
                    val_acc = float(torch.sum(torch.eq(val_res["pred_label"], val_dataset.labels))) / len(val_dataset)
                    logging.info(f"Validation accuracy: {val_acc: .4f}")
                    is_better = val_acc > best_metric
                    val_metric = val_acc

                if is_better:
                    logging.info("New best! Saving checkpoint")
                    best_metric = val_metric
                    no_increase = 0
                    self.save_pretrained(self.model_save_path)
                else:
                    no_increase += 1

                if no_increase == self.early_stopping_tol:
                    logging.info(f"Stopping early after validation metric did not improve for "
                                 f"{self.early_stopping_tol} rounds")
                    stop_early = True
                    break

            if stop_early:
                break

        logging.info(f"Training took {time() - train_start:.4f}s")
