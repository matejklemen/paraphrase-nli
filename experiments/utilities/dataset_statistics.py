import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import stanza

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", help="Path to the directory of an experiment",
                    default="/home/matej/Documents/paraphrase-nli/experiments/ASSIN/PARAPHRASE_GENERATION/assin-bert-large-portuguese-cased-argmax-mcd10")
parser.add_argument("--stanza_lang", help="Language of the data in the experiment - used to load correct tokenizer",
                    default="pt")

if __name__ == "__main__":
    args = parser.parse_args()
    path_to_results = os.path.join(args.experiment_dir, "all_paraphrases.csv")
    experiment_name = path_to_results.split(os.path.sep)[-2]

    df = pd.read_csv(path_to_results)
    print(f"Loaded {df.shape[0]} sequences...")

    is_identical = df["sequence1"].apply(lambda curr_str: curr_str.lower().strip()) == df["sequence2"].apply(lambda curr_str: curr_str.lower().strip())
    df = df.loc[np.logical_not(is_identical)].reset_index(drop=True)
    print(f"Keeping {df.shape[0]} sequences after filtering paraphrases with identical sentences...")

    s1 = df["sequence1"].tolist()
    s2 = df["sequence2"].tolist()

    try:
        nlp = stanza.Pipeline(lang=args.stanza_lang, processors="tokenize", use_gpu=False, tokenize_no_ssplit=True)
    except stanza.pipeline.core.LanguageNotDownloadedError:
        stanza.download(args.stanza_lang, processors="tokenize")
        nlp = stanza.Pipeline(lang=args.stanza_lang, processors="tokenize", use_gpu=False, tokenize_no_ssplit=True)

    tokenized_s1 = []
    tokenized_s2 = []

    STANZA_BATCH_SIZE = 1024
    for idx_subset in range((df.shape[0] + STANZA_BATCH_SIZE - 1) // STANZA_BATCH_SIZE):
        s, e = idx_subset * STANZA_BATCH_SIZE, (1 + idx_subset) * STANZA_BATCH_SIZE
        for tok_s1, tok_s2 in zip(nlp("\n\n".join(s1[s: e])).sentences,
                                  nlp("\n\n".join(s2[s: e])).sentences):
            tokenized_s1.append([token.words[0].text for token in tok_s1.tokens])
            tokenized_s2.append([token.words[0].text for token in tok_s2.tokens])

    num_tokens_by_seq = [[], []]
    prop_overlapping = []
    len_diff = []

    for curr_s1, curr_s2 in zip(tokenized_s1, tokenized_s2):
        num_tokens_by_seq[0].append(len(curr_s1))
        num_tokens_by_seq[1].append(len(curr_s2))
        len_diff.append(abs(len(curr_s1) - len(curr_s2)))

        num_overlapping_tokens = sum((Counter(map(lambda curr_str: curr_str.lower(), curr_s1)) &
                                      Counter(map(lambda curr_str: curr_str.lower(), curr_s2))).values())
        num_total_tokens = len(curr_s1) + len(curr_s2)
        prop_overlapping.append(num_overlapping_tokens / num_total_tokens)

    print(f"'{experiment_name}' ({df.shape[0]} pairs)")
    print(f"Sequence length [MED]: s1 = {np.median(num_tokens_by_seq[0]):.2f}, s2 = {np.median(num_tokens_by_seq[1]):.2f}")
    print(f"Word overlap [MED]: {np.median(prop_overlapping) * 100.0:.2f}%, [MAX]: {prop_overlapping[-1] * 100.0:.2f}%")
    print(f"Length diff. between pairs [MED]: {np.median(len_diff):.2f}")



















