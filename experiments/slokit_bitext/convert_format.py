from random import random

import numpy as np
import pandas as pd
from tqdm import trange

if __name__ == "__main__":
	file_path = "/home/matej/Documents/slo-kit/data/newsela-translated.jsonl"
	df = pd.read_json(file_path, orient="records", lines=True)

	data = {"doc_id": [], "src_grade": [], "tgt_grade": [], "pair_id": [],
			"idx_orig_row": [], "orig_sent_en": [], "orig_sent_sl": [], "type": [],
			"sentence1": [], "sentence2": [], "is_translation": []}
	for idx_ex in trange(df.shape[0]):
		curr_ex = df.iloc[idx_ex]

		data["doc_id"].append(curr_ex["doc_id"])
		data["src_grade"].append(curr_ex["src_grade"])
		data["tgt_grade"].append(curr_ex["tgt_grade"])
		data["pair_id"].append(curr_ex["pair_id"])
		data["idx_orig_row"].append(idx_ex)
		data["type"].append("src_")
		data["is_translation"].append("translation")

		# Original order: en->sl
		if random() < 0.5:
			data["sentence1"].append(curr_ex["src_sent_en"])
			data["sentence2"].append(curr_ex["src_sent_sl"])
			data["orig_sent_en"].append(curr_ex["src_sent_en"])
			data["orig_sent_sl"].append(curr_ex["src_sent_sl"])
		# 50% chance to reverse order: sl->en
		else:
			data["sentence1"].append(curr_ex["src_sent_sl"])
			data["sentence2"].append(curr_ex["src_sent_en"])
			data["orig_sent_en"].append(curr_ex["src_sent_en"])
			data["orig_sent_sl"].append(curr_ex["src_sent_sl"])

		data["doc_id"].append(curr_ex["doc_id"])
		data["src_grade"].append(curr_ex["src_grade"])
		data["tgt_grade"].append(curr_ex["tgt_grade"])
		data["pair_id"].append(curr_ex["pair_id"])
		data["idx_orig_row"].append(idx_ex)
		data["type"].append("tgt_")
		data["is_translation"].append("translation")

		# Original order: en->sl
		if random() < 0.5:
			data["sentence1"].append(curr_ex["tgt_sent_en"])
			data["sentence2"].append(curr_ex["tgt_sent_sl"])
			data["orig_sent_en"].append(curr_ex["tgt_sent_en"])
			data["orig_sent_sl"].append(curr_ex["tgt_sent_sl"])
		# 50% chance to reverse order: sl->en
		else:
			data["sentence1"].append(curr_ex["tgt_sent_sl"])
			data["sentence2"].append(curr_ex["tgt_sent_en"])
			data["orig_sent_en"].append(curr_ex["tgt_sent_en"])
			data["orig_sent_sl"].append(curr_ex["tgt_sent_sl"])

	pd_data = pd.DataFrame(data)
	pd_data.to_csv("slokit-parallel-candidates.tsv", sep="\t", index=False)
	print(f"Wrote {pd_data.shape[0]} candidates")

	rand_neg_indices = np.random.permutation(df.shape[0])
	# Create random negative examples
	for idx_ex in trange(df.shape[0]):
		curr_ex = df.iloc[idx_ex]
		idx_neg_ex = rand_neg_indices[idx_ex]
		neg_ex = df.iloc[idx_neg_ex]

		data["doc_id"].append(curr_ex["doc_id"])
		data["src_grade"].append("N/A")
		data["tgt_grade"].append("N/A")
		data["pair_id"].append("N/A")
		data["idx_orig_row"].append(-1)
		data["type"].append("dummy_")
		data["is_translation"].append("not_translation")

		# Original order: en->sl
		if random() < 0.5:
			data["sentence1"].append(curr_ex["src_sent_en"])
			data["sentence2"].append(neg_ex["src_sent_sl"])
			data["orig_sent_en"].append(curr_ex["src_sent_en"])
			data["orig_sent_sl"].append(neg_ex["src_sent_sl"])
		# 50% chance to reverse order: sl->en
		else:
			data["sentence1"].append(neg_ex["src_sent_sl"])
			data["sentence2"].append(curr_ex["src_sent_en"])
			data["orig_sent_en"].append(curr_ex["src_sent_en"])
			data["orig_sent_sl"].append(neg_ex["src_sent_sl"])

		data["doc_id"].append(curr_ex["doc_id"])
		data["src_grade"].append("N/A")
		data["tgt_grade"].append("N/A")
		data["pair_id"].append("N/A")
		data["idx_orig_row"].append(-1)
		data["type"].append("dummy_")
		data["is_translation"].append("not_translation")

		# Original order: en->sl
		if random() < 0.5:
			data["sentence1"].append(curr_ex["tgt_sent_en"])
			data["sentence2"].append(neg_ex["tgt_sent_sl"])
			data["orig_sent_en"].append(curr_ex["tgt_sent_en"])
			data["orig_sent_sl"].append(neg_ex["tgt_sent_sl"])
		# 50% chance to reverse order: sl->en
		else:
			data["sentence1"].append(neg_ex["tgt_sent_sl"])
			data["sentence2"].append(curr_ex["tgt_sent_en"])
			data["orig_sent_en"].append(curr_ex["tgt_sent_en"])
			data["orig_sent_sl"].append(neg_ex["tgt_sent_sl"])

	pd_data = pd.DataFrame(data)

	rand_indices = np.random.permutation(pd_data.shape[0])
	train_indices = rand_indices[:-20_000]
	dev_indices = rand_indices[-20_000: -10_000]
	test_indices = rand_indices[-10_000:]
	print(f"Created {pd_data.shape[0]} total examples for slokit bitext filtering:\n"
		  f"\t-{train_indices.shape[0]} train examples,\n"
		  f"\t-{dev_indices.shape[0]} dev examples,\n"
		  f"\t-{test_indices.shape[0]} test examples")

	pd_data.iloc[train_indices].to_csv("train.tsv", sep="\t", index=False)
	pd_data.iloc[dev_indices].to_csv("dev.tsv", sep="\t", index=False)
	pd_data.iloc[test_indices].to_csv("test.tsv", sep="\t", index=False)
