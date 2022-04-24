import logging
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import IterableDataset

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


class WMT14TransformersStreamingDataset(IterableDataset):
    def __init__(self, path: str, tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False, chunksize: int = 100_000):
        super().__init__()

        self.path = path
        self.chunksize = chunksize
        self.tokenizer = tokenizer
        self.reverse_order = reverse_order

        self.optional_kwargs = {}
        if return_tensors is not None:
            self.optional_kwargs["return_tensors"] = return_tensors

        if max_length is not None:
            self.optional_kwargs["max_length"] = max_length
            self.optional_kwargs["padding"] = "max_length"
            self.optional_kwargs["truncation"] = "longest_first"

    def __iter__(self):
        for curr_chunk in pd.read_csv(self.path, chunksize=self.chunksize, sep="\t", lineterminator="\n"):
            s1 = curr_chunk["sentence1"].tolist()
            s2 = curr_chunk["sentence2"].tolist()
            if self.reverse_order:
                s1, s2 = s2, s1

            encoded = self.tokenizer.batch_encode_plus(list(zip(s1, s2)), **self.optional_kwargs)
            if "is_translation" in curr_chunk.columns:
                labels = curr_chunk["is_translation"].tolist()
                if self.optional_kwargs.get("return_tensors", None) is not None:
                    labels = torch.tensor(labels)

                encoded["labels"] = labels

            for i in range(curr_chunk.shape[0]):
                yield {_k: _v[i] for _k, _v in encoded.items()}


if __name__ == "__main__":
    from transformers import XLMRobertaTokenizerFast
    from torch.utils.data import IterableDataset, DataLoader
    data_path = "/home/matej/Documents/paraphrase-nli/experiments/WMT14_bitext/id_wmt14_hi-en_bitext_train_mixeddir.tsv"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    data = WMT14TransformersStreamingDataset(data_path, tokenizer=tokenizer, max_length=30, return_tensors="pt",
                                             chunksize=10_000)
    for ex in DataLoader(data, batch_size=4):
        print(ex)
