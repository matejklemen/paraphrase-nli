import argparse
import logging
import os
import sys
from collections import Counter
from copy import deepcopy

import numpy as np
import stanza
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import BertTokenizerFast

from src.data.paraphrase import QQPTransformersDataset

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str,
                    default="debug")
parser.add_argument("--train_path", type=str,
                    default="/home/matej/Documents/data/qqp/train.tsv")
parser.add_argument("--dev_path", type=str,
                    default="/home/matej/Documents/data/qqp/dev.tsv")
parser.add_argument("--use_cpu", action="store_true")


if __name__ == "__main__":
    """ A simple overlap baseline to compare the baseline performance across datasets. """
    args = parser.parse_args()
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    dummy_tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    train_set = QQPTransformersDataset(args.train_path, tokenizer=dummy_tokenizer)
    dev_set = QQPTransformersDataset(args.dev_path, tokenizer=dummy_tokenizer)

    indices = np.random.permutation(len(dev_set))
    dev_indices = indices[: int(0.5 * len(indices))]
    test_indices = indices[int(0.5 * len(indices)):]

    # Instantiate dummy set, override with actual data
    test_set = deepcopy(dev_set)

    # NOTE: order important here! First copy to test set, then fix dev set itself
    for curr_set, curr_indices in [(test_set, test_indices), (dev_set, dev_indices)]:
        curr_set.pair_id = [dev_set.pair_id[_i] for _i in curr_indices]

        curr_set.qid1 = [dev_set.qid1[_i] for _i in curr_indices]
        curr_set.qid2 = [dev_set.qid2[_i] for _i in curr_indices]

        curr_set.seq1 = [dev_set.seq1[_i] for _i in curr_indices]
        curr_set.seq2 = [dev_set.seq2[_i] for _i in curr_indices]

        curr_set.num_examples = len(curr_set.pair_id)

    logging.info(f"Loaded {len(train_set)} training examples, "
                 f"{len(dev_set)} dev examples and "
                 f"{len(test_set)} test examples")

    try:
        nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=(not args.use_cpu), tokenize_no_ssplit=True)
    except stanza.pipeline.core.LanguageNotDownloadedError:
        stanza.download("en", processors="tokenize")
        nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=(not args.use_cpu), tokenize_no_ssplit=True)

    train_overlaps = []
    STANZA_BATCH_SIZE = 1024
    num_train_batches = (len(train_set) + STANZA_BATCH_SIZE - 1) // STANZA_BATCH_SIZE
    for idx_subset in tqdm(range(num_train_batches), total=num_train_batches):
        s, e = idx_subset * STANZA_BATCH_SIZE, (1 + idx_subset) * STANZA_BATCH_SIZE
        for tok_s1, tok_s2 in zip(nlp("\n\n".join(train_set.seq1[s: e])).sentences,
                                  nlp("\n\n".join(train_set.seq2[s: e])).sentences):
            c1 = Counter(map(lambda _s: _s.text.lower(), tok_s1.words))
            c2 = Counter(map(lambda _s: _s.text.lower(), tok_s2.words))
            num_overlapping_tokens = sum((c1 & c2).values())
            num_total_tokens = len(tok_s1.words) + len(tok_s2.words)

            train_overlaps.append([num_overlapping_tokens / num_total_tokens])

    test_overlaps = []
    num_test_batches = (len(test_set) + STANZA_BATCH_SIZE - 1) // STANZA_BATCH_SIZE
    for idx_subset in tqdm(range(num_test_batches), total=num_test_batches):
        s, e = idx_subset * STANZA_BATCH_SIZE, (1 + idx_subset) * STANZA_BATCH_SIZE
        for tok_s1, tok_s2 in zip(nlp("\n\n".join(test_set.seq1[s: e])).sentences,
                                  nlp("\n\n".join(test_set.seq2[s: e])).sentences):
            c1 = Counter(map(lambda _s: _s.text.lower(), tok_s1.words))
            c2 = Counter(map(lambda _s: _s.text.lower(), tok_s2.words))
            num_overlapping_tokens = sum((c1 & c2).values())
            num_total_tokens = len(tok_s1.words) + len(tok_s2.words)

            test_overlaps.append([num_overlapping_tokens / num_total_tokens])

    model = LogisticRegression()
    model.fit(np.array(train_overlaps), np.array(train_set.labels))

    preds = model.predict(np.array(test_overlaps))
    f1 = f1_score(y_true=np.array(test_set.labels), y_pred=preds, average="macro")
    acc = accuracy_score(y_true=np.array(test_set.labels), y_pred=preds)
    logging.info(f"Accuracy: {acc: .2f}, F1: {f1: .2f}")



