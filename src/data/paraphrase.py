import csv
import logging
from io import StringIO
from typing import Union, Iterable, Optional
from warnings import warn

import torch

from src.data.nli import TransformersSeqPairDataset

""" TODO: PIT2015? """


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
                 reverse_order: Optional[bool] = False):
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

        if reverse_order:
            self.sid1, self.sid2 = self.sid2, self.sid1
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
