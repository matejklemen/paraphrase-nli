import logging
from typing import Optional

import pandas as pd
import torch

from src.data.nli import TransformersSeqPairDataset


class KASTransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: str, tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False):
        data = pd.read_csv(path, sep="\t")

        self.seq1 = data["sentence1"].tolist()
        self.seq2 = data["sentence2"].tolist()
        valid_label = data["is_translation"].tolist()

        if reverse_order:
            self.seq1, self.seq2 = self.seq2, self.seq1

        self.label_names = ["not_translation", "translation"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Sanity check: expecting binary task
        assert len(set(valid_label)) == 2

        optional_kwargs = {}
        if return_tensors is not None:
            valid_label = torch.tensor(valid_label)
            optional_kwargs["return_tensors"] = "pt"

        if max_length is not None:
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = tokenizer.batch_encode_plus(list(zip(self.seq1, self.seq2)), **optional_kwargs)
        encoded["labels"] = valid_label

        super().__init__(**encoded)


class RSDO4TransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: str, tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False):
        data = pd.read_csv(path, sep="\t", keep_default_na=False)

        self.seq1 = data["sentence1"].tolist()
        self.seq2 = data["sentence2"].tolist()
        valid_label = data["is_translation"].tolist()

        if reverse_order:
            self.seq1, self.seq2 = self.seq2, self.seq1

        self.label_names = ["not_translation", "translation"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Sanity check: expecting binary task
        assert len(set(valid_label)) == 2

        optional_kwargs = {}
        if return_tensors is not None:
            valid_label = torch.tensor(valid_label)
            optional_kwargs["return_tensors"] = "pt"

        if max_length is not None:
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = tokenizer.batch_encode_plus(list(zip(self.seq1, self.seq2)), **optional_kwargs)
        encoded["labels"] = valid_label

        super().__init__(**encoded)


class WMT14TransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: str, tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False,
                 nrows: Optional[int] = None):
        if nrows is not None:
            logging.warning(f"Only using first {nrows} rows...")

        data = pd.read_csv(path, sep="\t", lineterminator="\n", nrows=nrows)

        self.seq1 = data["sentence1"].tolist()
        self.seq2 = data["sentence2"].tolist()
        valid_label = data["is_translation"].tolist()

        if reverse_order:
            self.seq1, self.seq2 = self.seq2, self.seq1

        self.label_names = ["not_translation", "translation"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Sanity check: expecting binary task
        assert len(set(valid_label)) == 2

        optional_kwargs = {}
        if return_tensors is not None:
            valid_label = torch.tensor(valid_label)
            optional_kwargs["return_tensors"] = "pt"

        if max_length is not None:
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = tokenizer.batch_encode_plus(list(zip(self.seq1, self.seq2)), **optional_kwargs)
        encoded["labels"] = valid_label

        super().__init__(**encoded)

