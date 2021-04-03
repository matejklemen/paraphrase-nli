from typing import Optional

import datasets
import torch
from torch.utils.data import Dataset

""" 
This file contains NLI-related dataset constructors.

TODO: ESNLI? -> explanations are paraphrases by design
TODO: FEVER?
"""


class TransformersSeqPairDataset(Dataset):
    def __init__(self, **kwargs):
        self.valid_attrs = []
        self.num_examples = 0
        for attr, values in kwargs.items():
            self.valid_attrs.append(attr)
            setattr(self, attr, values)
            self.num_examples = len(values)

    def __getitem__(self, item):
        return {k: getattr(self, k)[item] for k in self.valid_attrs}

    def __len__(self):
        return self.num_examples


class SNLITransformersDataset(TransformersSeqPairDataset):
    def __init__(self, split: str, tokenizer, max_length: Optional[int] = None, return_tensors: Optional[str] = None):
        dataset = datasets.load_dataset("snli", split=split)
        all_hypothesis = dataset["hypothesis"]
        all_premise = dataset["premise"]
        all_label = dataset["label"]

        self.label_names = dataset.features["label"].names
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Examples that have a valid label (!= -1)
        valid_indices = [_i for _i in range(len(dataset)) if all_label[_i] != -1]

        self.str_premise = [all_premise[_i] for _i in valid_indices]
        self.str_hypothesis = [all_hypothesis[_i] for _i in valid_indices]
        valid_label = [all_label[_i] for _i in valid_indices]

        optional_kwargs = {}
        if return_tensors is not None:
            valid_label = torch.tensor(valid_label)
            optional_kwargs["return_tensors"] = "pt"

        if max_length is not None:
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = tokenizer.batch_encode_plus(list(zip(self.str_premise, self.str_hypothesis)), **optional_kwargs)
        encoded["labels"] = valid_label

        super().__init__(**encoded)


class MultiNLITransformersDataset(TransformersSeqPairDataset):
    def __init__(self, split: str, tokenizer, max_length: Optional[int] = None, return_tensors: Optional[str] = None):
        dataset = datasets.load_dataset("multi_nli", split=split)
        all_pair_ids = dataset["pairID"]
        all_genres = dataset["genre"]
        all_hypothesis = dataset["hypothesis"]
        all_premise = dataset["premise"]
        all_label = dataset["label"]

        self.label_names = dataset.features["label"].names
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Examples that have a valid label (!= -1)
        valid_indices = [_i for _i in range(len(dataset)) if all_label[_i] != -1]

        self.pair_ids = [all_pair_ids[_i] for _i in valid_indices]
        self.str_premise = [all_premise[_i] for _i in valid_indices]
        self.str_hypothesis = [all_hypothesis[_i] for _i in valid_indices]
        self.genre = [all_genres[_i] for _i in valid_indices]
        valid_label = [all_label[_i] for _i in valid_indices]

        optional_kwargs = {}
        if return_tensors is not None:
            valid_label = torch.tensor(valid_label)
            optional_kwargs["return_tensors"] = "pt"

        if max_length is not None:
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = tokenizer.batch_encode_plus(list(zip(self.str_premise, self.str_hypothesis)), **optional_kwargs)
        encoded["labels"] = valid_label

        super().__init__(**encoded)


class XNLITransformersDataset(TransformersSeqPairDataset):
    def __init__(self, lang: str, split: str, tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None):
        dataset = datasets.load_dataset("xnli", lang, split=split)
        all_hypothesis = dataset["hypothesis"]
        all_premise = dataset["premise"]
        all_label = dataset["label"]

        self.label_names = dataset.features["label"].names
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Examples that have a valid label (!= -1)
        valid_indices = [_i for _i in range(len(dataset)) if all_label[_i] != -1]

        self.str_premise = [all_premise[_i] for _i in valid_indices]
        self.str_hypothesis = [all_hypothesis[_i] for _i in valid_indices]
        valid_label = [all_label[_i] for _i in valid_indices]

        optional_kwargs = {}
        if return_tensors is not None:
            valid_label = torch.tensor(valid_label)
            optional_kwargs["return_tensors"] = "pt"

        if max_length is not None:
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = tokenizer.batch_encode_plus(list(zip(self.str_premise, self.str_hypothesis)), **optional_kwargs)
        encoded["labels"] = valid_label

        super().__init__(**encoded)


if __name__ == "__main__":
    from transformers import BertTokenizerFast, RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    dataset = XNLITransformersDataset("de", split="test", tokenizer=tokenizer)

    print(len(dataset))
