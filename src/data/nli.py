import itertools
from typing import Optional, List, Union, Iterable

import datasets
import torch
from torch.utils.data import Dataset
import pandas as pd

""" 
This file contains NLI-related dataset constructors.

TODO: ESNLI? -> explanations are paraphrases by design
TODO: ANLI?
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
    def __init__(self, split: Union[str, Iterable[str]], tokenizer, max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 custom_label_names: Optional[List[str]] = None):
        _split = (split,) if isinstance(split, str) else split

        datasets_list = [datasets.load_dataset("snli", split=curr_split) for curr_split in _split]
        all_hypothesis = list(itertools.chain(*[curr_dataset["hypothesis"] for curr_dataset in datasets_list]))
        all_premise = list(itertools.chain(*[curr_dataset["premise"] for curr_dataset in datasets_list]))
        all_label = list(itertools.chain(*[curr_dataset["label"] for curr_dataset in datasets_list]))

        if custom_label_names is None:
            self.label_names = datasets_list[0].features["label"].names
        else:
            self.label_names = custom_label_names

        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Examples that have a valid label (!= -1)
        valid_indices = [_i for _i in range(len(all_label)) if all_label[_i] != -1]

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
    def __init__(self, split: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 custom_label_names: Optional[List[str]] = None):
        _split = (split,) if isinstance(split, str) else split

        datasets_list = [datasets.load_dataset("multi_nli", split=curr_split) for curr_split in _split]
        all_pair_ids = list(itertools.chain(*[curr_dataset["pairID"] for curr_dataset in datasets_list]))
        all_genres = list(itertools.chain(*[curr_dataset["genre"] for curr_dataset in datasets_list]))
        all_hypothesis = list(itertools.chain(*[curr_dataset["hypothesis"] for curr_dataset in datasets_list]))
        all_premise = list(itertools.chain(*[curr_dataset["premise"] for curr_dataset in datasets_list]))
        all_label = list(itertools.chain(*[curr_dataset["label"] for curr_dataset in datasets_list]))

        if custom_label_names is None:
            self.label_names = datasets_list[0].features["label"].names
        else:
            self.label_names = custom_label_names

        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Examples that have a valid label (!= -1)
        valid_indices = [_i for _i in range(len(all_label)) if all_label[_i] != -1]

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
    def __init__(self, lang: Union[str, Iterable[str]], split: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 custom_label_names: Optional[List[str]] = None):
        _lang = (lang,) if isinstance(lang, str) else lang
        _split = (split,) if isinstance(split, str) else split

        datasets_list = [datasets.load_dataset("xnli", curr_lang, split=curr_split)
                         for curr_lang, curr_split in zip(_lang, _split)]
        all_hypothesis = list(itertools.chain(*[curr_dataset["hypothesis"] for curr_dataset in datasets_list]))
        all_premise = list(itertools.chain(*[curr_dataset["premise"] for curr_dataset in datasets_list]))
        all_label = list(itertools.chain(*[curr_dataset["label"] for curr_dataset in datasets_list]))

        if custom_label_names is None:
            self.label_names = datasets_list[0].features["label"].names
        else:
            self.label_names = custom_label_names

        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        # Examples that have a valid label (!= -1)
        valid_indices = [_i for _i in range(len(all_label)) if all_label[_i] != -1]

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


class RTETransformersDataset(TransformersSeqPairDataset):
    def __init__(self, path: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None):
        _path = (path,) if isinstance(path, str) else path
        df = pd.concat([pd.read_csv(curr_path) for curr_path in _path]).reset_index(drop=True)

        self.label_names = ["not_entailment", "entailment"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        self.str_premise = df["premise"].tolist()
        self.str_hypothesis = df["hypothesis"].tolist()

        valid_label = df["idx"].tolist()

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
    dataset = MultiNLITransformersDataset(("validation_matched", "validation_mismatched"), tokenizer=tokenizer)

    print(len(dataset))
