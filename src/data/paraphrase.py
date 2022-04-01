import csv
import logging
from collections import Counter
from io import StringIO
from typing import Union, Iterable, Optional
from warnings import warn

import pandas as pd
import torch

from src.data.nli import TransformersSeqPairDataset


class QQPTransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False):
        _path = (path,) if isinstance(path, str) else path

        self.pair_id = []
        self.qid1 = []
        self.qid2 = []

        self.seq1 = []
        self.seq2 = []

        valid_label = []

        header = None
        for curr_path in _path:
            with open(curr_path, "r", encoding="utf-8") as f:
                data = list(map(lambda s: s.strip(), f.readlines()))

            header = data[0].split("\t")
            assert ("is_duplicate" in header and len(header) == 6) or \
                   ("is_duplicate" not in header and len(header) == 5)

            num_errs = 0
            for i, curr_row in enumerate(data[1:], start=1):
                fields = list(csv.reader(StringIO(curr_row), delimiter="\t"))[0]

                if len(fields) != len(header):
                    num_errs += 1
                    continue

                self.pair_id.append(int(fields[0]))
                self.qid1.append(int(fields[1]))
                self.qid2.append(int(fields[2]))
                self.seq1.append(fields[3])
                self.seq2.append(fields[4])

                if len(fields) == 6:
                    valid_label.append(int(fields[-1]))

            logging.info(f"'{curr_path}': skipped {num_errs} rows due to formatting errors")

        if reverse_order:
            self.qid1, self.qid2 = self.qid2, self.qid1
            self.seq1, self.seq2 = self.seq2, self.seq1

        self.label_names = ["not_paraphrase", "paraphrase"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        if "is_duplicate" not in header:
            warn(f"No labels present in file - setting all labels to 0, so you should ignore metrics based on these")
            valid_label = [0] * len(self.seq1)
        else:
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


class MRPCTransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False, balance: Optional[bool] = False):
        _path = (path,) if isinstance(path, str) else path

        self.sid1 = []
        self.sid2 = []

        self.seq1 = []
        self.seq2 = []

        valid_label = []

        for curr_path in _path:
            with open(curr_path, "r", encoding="utf-8-sig") as f:
                data = list(map(lambda s: s.strip(), f.readlines()))

            header = data[0].split("\t")
            assert len(header) == 5

            num_errs = 0
            for i, curr_row in enumerate(data[1:], start=1):
                fields = list(csv.reader(StringIO(curr_row), delimiter="\t", quoting=csv.QUOTE_NONE))[0]

                if len(fields) != len(header):
                    num_errs += 1
                    continue

                valid_label.append(int(fields[0]))
                self.sid1.append(int(fields[1]))
                self.sid2.append(int(fields[2]))
                self.seq1.append(fields[3])
                self.seq2.append(fields[4])

            logging.info(f"'{curr_path}': skipped {num_errs} rows due to formatting errors")

        # Sanity check: expecting binary task
        assert len(set(valid_label)) == 2

        if balance:
            label_count = Counter(valid_label)
            assert label_count[1] > label_count[0]

            _valid_label = torch.tensor(valid_label)
            # Keep all examples except those corresponding to excess positive labels
            keep_mask = torch.ones_like(_valid_label, dtype=torch.bool)
            positive_indices = torch.flatten(torch.nonzero(_valid_label, as_tuple=False))
            positive_indices = positive_indices[torch.randperm(positive_indices.shape[0])]
            keep_mask[positive_indices[label_count[0]:]] = False

            keep_indices = torch.flatten(torch.nonzero(keep_mask, as_tuple=False)).tolist()
            self.sid1 = [self.sid1[_i] for _i in keep_indices]
            self.sid2 = [self.sid2[_i] for _i in keep_indices]

            self.seq1 = [self.seq1[_i] for _i in keep_indices]
            self.seq2 = [self.seq2[_i] for _i in keep_indices]

            valid_label = [valid_label[_i] for _i in keep_indices]
            logging.info(f"Balancing dataset:\n"
                         f"\t Before: {label_count}\n"
                         f"\t After: {Counter(valid_label)}")

        if reverse_order:
            self.sid1, self.sid2 = self.sid2, self.sid1
            self.seq1, self.seq2 = self.seq2, self.seq1

        self.label_names = ["not_paraphrase", "paraphrase"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

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


class MSCOCOTransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: str, tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False):
        data = pd.read_csv(path, sep="\t")

        self.imid1 = data["sentence1_image_id"].tolist()
        self.imid2 = data["sentence2_image_id"].tolist()

        self.seq1 = data["sentence1"].tolist()
        self.seq2 = data["sentence2"].tolist()
        valid_label = data["is_paraphrase"].tolist()

        if reverse_order:
            self.imid1, self.imid2 = self.imid2, self.imid1
            self.seq1, self.seq2 = self.seq2, self.seq1

        self.label_names = ["not_paraphrase", "paraphrase"]
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


class TapacoTransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: str, tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 reverse_order: Optional[bool] = False):
        data = pd.read_csv(path, sep="\t")

        self.seq1 = data["sentence1"].tolist()
        self.seq2 = data["sentence2"].tolist()
        self.lang = data["language"].tolist()
        valid_label = data["is_paraphrase"].tolist()

        if reverse_order:
            self.seq1, self.seq2 = self.seq2, self.seq1

        self.label_names = ["not_paraphrase", "paraphrase"]
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


if __name__ == "__main__":
    from transformers import RobertaTokenizerFast, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    data = TapacoTransformersDataset("/home/matej/Documents/paraphrase-nli/experiments/TaPaCo_para/en/id_tapaco_en_train.tsv",
                                     tokenizer=tokenizer)

