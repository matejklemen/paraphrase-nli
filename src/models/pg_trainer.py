import json
import logging
import os
from time import time
from typing import Optional, List

import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class AutoregressivePGTrainer:
    def __init__(self, model_dir, pretrained_model_name_or_path, tokenizer_path, pred_strategy="argmax",
                 batch_size=24, learning_rate=6.25e-5, validate_every_n_steps=5_000, early_stopping_tol=5,
                 use_mcd: Optional[bool] = False, device="cuda"):
        self.model_save_path = model_dir

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validate_every_n_steps = validate_every_n_steps
        self.early_stopping_tol = early_stopping_tol

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device set to 'cuda' but no CUDA device is available!")
        self.device_str = device
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")

        self.pred_strategy = pred_strategy
        self.use_mcd = use_mcd

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path,
                                                          return_dict=True).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def save_pretrained(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(os.path.join(save_dir, "trainer_config.json")):
            with open(os.path.join(save_dir, "trainer_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "model_dir": save_dir,
                    "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
                    "tokenizer_path": self.tokenizer_path,
                    "pred_strategy": self.pred_strategy,
                    "use_mcd": self.use_mcd,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "validate_every_n_steps": self.validate_every_n_steps,
                    "early_stopping_tol": self.early_stopping_tol,
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

        instance = AutoregressivePGTrainer(**pretrained_config)
        return instance

    def train(self, train_dataset, eval_mode=False):
        if not eval_mode:
            self.model.train()

        if eval_mode:
            def after_model_forward(_loss):
                return float(_loss)
        else:
            def after_model_forward(_loss):
                _loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                return float(_loss)

        num_batches = (len(train_dataset) + self.batch_size - 1) // self.batch_size
        train_loss = 0.0
        for curr_batch in tqdm(DataLoader(train_dataset, shuffle=False, batch_size=self.batch_size),
                               total=num_batches):
            res = self.model(**{k: v.to(self.device) for k, v in curr_batch.items()})
            train_loss += after_model_forward(res["loss"])

        return {"train_loss": train_loss}

    @torch.no_grad()
    def evaluate(self, val_dataset):
        if self.use_mcd:
            self.model.train()
        else:
            self.model.eval()

        res = self.train(val_dataset, eval_mode=True)
        res["eval_loss"] = res.pop("train_loss")

        return res

    @torch.no_grad()
    def generate(self, prepr_prompts: List, **generation_kwargs):
        if self.use_mcd:
            self.model.train()
        else:
            self.model.eval()

        max_generated_length = generation_kwargs["max_seq_len"]
        curr_strat = generation_kwargs.get("strategy", {})  # by default, greedy decoding strategy

        pred_para = []
        for idx_example in tqdm(range(len(prepr_prompts)), total=len(prepr_prompts)):
            curr_prompt = prepr_prompts[idx_example]
            curr_encoded = self.tokenizer.batch_encode_plus([curr_prompt], return_tensors="pt")
            take_from_idx = len(curr_encoded["input_ids"][0])
            eff_max_len = len(curr_encoded["input_ids"][0]) + max_generated_length

            curr_output = self.model.generate(curr_encoded["input_ids"].to(self.device),
                                              pad_token_id=self.tokenizer.pad_token_id,
                                              eos_token_id=self.tokenizer.eos_token_id,
                                              max_length=eff_max_len, **curr_strat)
            pred_para.append(self.tokenizer.decode(curr_output[0, take_from_idx:].cpu(), skip_special_tokens=True))

        return pred_para

    def run(self, train_dataset, val_dataset, num_epochs):
        best_metric, no_increase = float("inf"), 0
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
                train_res = self.train(curr_subset)
                train_loss += train_res["train_loss"]
                nb += (len(curr_subset) / self.batch_size)
                logging.info(f"Training loss = {train_loss / nb: .4f}")

                if val_dataset is None or len(curr_subset) < self.validate_every_n_steps // 2:
                    logging.info(f"Skipping validation after training on a small training subset "
                                 f"({len(curr_subset)} < {self.validate_every_n_steps // 2} examples)")
                    continue

                val_res = self.evaluate(val_dataset)
                val_loss = val_res["eval_loss"]
                logging.info(f"Validation loss = {val_loss / (len(val_dataset) / self.batch_size): .4f}")

                if val_loss < best_metric:
                    logging.info("New best! Saving checkpoint")
                    best_metric = val_loss
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
