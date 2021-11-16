import itertools
from typing import Optional, List, Union, Iterable
from warnings import warn

import datasets
import torch
from torch.utils.data import Dataset
import pandas as pd


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
                 custom_label_names: Optional[List[str]] = None, binarize: Optional[bool] = False):
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

        if binarize:
            encoded["labels"] = (encoded["labels"] == self.label2idx["entailment"]).long()
            self.label_names = ["not_entailment", "entailment"]
            self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
            self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        super().__init__(**encoded)


class MultiNLITransformersDataset(TransformersSeqPairDataset):
    def __init__(self, split: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 custom_label_names: Optional[List[str]] = None, binarize: Optional[bool] = False):
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

        if binarize:
            encoded["labels"] = (encoded["labels"] == self.label2idx["entailment"]).long()
            self.label_names = ["not_entailment", "entailment"]
            self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
            self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        super().__init__(**encoded)


class XNLITransformersDataset(TransformersSeqPairDataset):
    def __init__(self, lang: Union[str, Iterable[str]], split: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 custom_label_names: Optional[List[str]] = None, binarize: Optional[bool] = False):
        _lang = (lang,) if isinstance(lang, str) else lang
        _split = (split,) if isinstance(split, str) else split
        self.tokenizer = tokenizer

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

        self.max_length = None
        if max_length is not None:
            self.max_length = max_length
            optional_kwargs["max_length"] = max_length
            optional_kwargs["padding"] = "max_length"
            optional_kwargs["truncation"] = "longest_first"

        encoded = self.tokenizer.batch_encode_plus(list(zip(self.str_premise, self.str_hypothesis)), **optional_kwargs)
        encoded["labels"] = valid_label

        if binarize:
            encoded["labels"] = (encoded["labels"] == self.label2idx["entailment"]).long()
            self.label_names = ["not_entailment", "entailment"]
            self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
            self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        super().__init__(**encoded)

    def override_data(self, new_seq1: List[str], new_seq2: List[str], new_labels: List[int],
                      new_label_set: List[str] = None):
        assert self.max_length is not None
        assert len(new_seq1) == len(new_seq2) == len(new_labels)

        if new_label_set is not None:
            self.label_names = new_label_set
            self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
            self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        self.str_premise = new_seq1
        self.str_hypothesis = new_seq2
        new_encoded = self.tokenizer.batch_encode_plus(list(zip(new_seq1, new_seq2)), max_length=self.max_length,
                                                       padding="max_length", truncation="longest_first",
                                                       return_tensors="pt")
        new_encoded["labels"] = torch.tensor(new_labels)

        assert all(attr_name in new_encoded for attr_name in self.valid_attrs)
        self.num_examples = len(new_seq1)
        for attr, values in new_encoded.items():
            setattr(self, attr, values)


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

        if "label" in df.columns:
            valid_label = list(map(lambda lbl: self.label2idx[lbl], df["label"].tolist()))
        else:
            warn(f"No labels present in file - setting all labels to 0, so you should ignore metrics based on these")
            valid_label = [0] * len(self.str_premise)

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


class SciTailTransformersDataset(TransformersSeqPairDataset):
    def __init__(self, split: Union[str, Iterable[str]], tokenizer, max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 custom_label_names: Optional[List[str]] = None, binarize: Optional[bool] = False):
        _split = (split,) if isinstance(split, str) else split

        datasets_list = [datasets.load_dataset("scitail", "tsv_format", split=curr_split) for curr_split in _split]
        all_hypothesis = list(itertools.chain(*[curr_dataset["hypothesis"] for curr_dataset in datasets_list]))
        all_premise = list(itertools.chain(*[curr_dataset["premise"] for curr_dataset in datasets_list]))
        all_label = list(itertools.chain(*[curr_dataset["label"] for curr_dataset in datasets_list]))

        if custom_label_names is None:
            self.label_names = ["neutral", "entails"]
        else:
            # SciTail is two-class NLI
            assert len(custom_label_names) == 2
            self.label_names = custom_label_names

        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}
        all_label = [self.label2idx.get(_lbl, -1) for _lbl in all_label]

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

        if binarize:
            # Leave the argument in for consistency though
            warn("'binarize' is an unused argument in SciTail as it is binary by default")

        super().__init__(**encoded)


class OCNLIDataset(TransformersSeqPairDataset):
    def __init__(self, path: Union[str, Iterable[str]], tokenizer,
                 max_length: Optional[int] = None, return_tensors: Optional[str] = None,
                 binarize: Optional[bool] = False):
        _path = (path,) if isinstance(path, str) else path
        df = pd.concat([pd.read_json(curr_path, lines=True) for curr_path in _path]).reset_index(drop=True)

        self.label_names = ["entailment", "neutral", "contradiction"]
        self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
        self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        self.str_premise = df["sentence1"].tolist()
        self.str_hypothesis = df["sentence2"].tolist()
        self.genre = df["genre"].tolist()

        if "label" in df.columns:
            valid_label = list(map(lambda lbl: self.label2idx.get(lbl, -1), df["label"].tolist()))
        else:
            warn(f"No labels present in file - setting all labels to 0, so you should ignore metrics based on these")
            valid_label = [0] * len(self.str_premise)

        # Examples that have a valid label (!= -1)
        valid_indices = [_i for _i in range(len(valid_label)) if valid_label[_i] != -1]

        self.str_premise = [self.str_premise[_i] for _i in valid_indices]
        self.str_hypothesis = [self.str_hypothesis[_i] for _i in valid_indices]
        self.genre = [self.genre[_i] for _i in valid_indices]
        valid_label = [valid_label[_i] for _i in valid_indices]

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

        if binarize:
            encoded["labels"] = (encoded["labels"] == self.label2idx["entailment"]).long()
            self.label_names = ["not_entailment", "entailment"]
            self.label2idx = {curr_label: i for i, curr_label in enumerate(self.label_names)}
            self.idx2label = {i: curr_label for curr_label, i in self.label2idx.items()}

        super().__init__(**encoded)


if __name__ == "__main__":
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # dataset = SciTailTransformersDataset("train", tokenizer)

    # hfl/chinese-roberta-wwm-ext
    # hfl/chinese-roberta-wwm-ext-large
    dataset = OCNLIDataset("/home/matej/Documents/data/ocnli/train.50k.json", tokenizer)
