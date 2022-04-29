from copy import deepcopy

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
	LANG1, LANG2 = "nl", "sl"
	data = load_dataset("yhavinga/ccmatrix", lang1=LANG1, lang2=LANG2)["train"]

	train_examples = {
		"sentence1": [],
		"sentence2": [],
		"is_translation": []
	}
	dev_examples = deepcopy(train_examples)
	test_examples = deepcopy(train_examples)

	num_examples = len(data)
	indices = np.random.permutation(num_examples)
	train_inds = indices[:-20_000]
	dev_inds = indices[-20_000: -10_000]
	test_inds = indices[-10_000:]

	for _idx, _idx_neg, _is_reverse in tqdm(zip(train_inds,
												np.random.permutation(train_inds),
												np.random.random(train_inds.shape[0]) < 0.5),
											total=len(train_inds)):
		curr_example = data[int(_idx)]
		random_negative_example = data[int(_idx_neg)]

		s1, s2 = curr_example["translation"][LANG1], curr_example["translation"][LANG2]
		if _is_reverse:
			s1, s2 = s2, s1

		train_examples["sentence1"].append(s1)
		train_examples["sentence2"].append(s2)
		train_examples["is_translation"].append(1)

		s1, s2 = curr_example["translation"][LANG1], random_negative_example["translation"][LANG2]
		if _is_reverse:
			s1, s2 = s2, s1

		train_examples["sentence1"].append(s1)
		train_examples["sentence2"].append(s2)
		train_examples["is_translation"].append(0)

	for _idx, _idx_neg, _is_reverse in tqdm(zip(dev_inds,
												np.random.permutation(dev_inds),
												np.random.random(dev_inds.shape[0]) < 0.5),
											total=len(dev_inds)):
		curr_example = data[int(_idx)]
		random_negative_example = data[int(_idx_neg)]

		s1, s2 = curr_example["translation"][LANG1], curr_example["translation"][LANG2]
		if _is_reverse:
			s1, s2 = s2, s1

		dev_examples["sentence1"].append(s1)
		dev_examples["sentence2"].append(s2)
		dev_examples["is_translation"].append(1)

		s1, s2 = curr_example["translation"][LANG1], random_negative_example["translation"][LANG2]
		if _is_reverse:
			s1, s2 = s2, s1

		dev_examples["sentence1"].append(s1)
		dev_examples["sentence2"].append(s2)
		dev_examples["is_translation"].append(0)

	for _idx, _idx_neg, _is_reverse in tqdm(zip(test_inds,
												np.random.permutation(test_inds),
												np.random.random(test_inds.shape[0]) < 0.5),
											total=len(test_inds)):
		curr_example = data[int(_idx)]
		random_negative_example = data[int(_idx_neg)]

		s1, s2 = curr_example["translation"][LANG1], curr_example["translation"][LANG2]
		if _is_reverse:
			s1, s2 = s2, s1

		test_examples["sentence1"].append(s1)
		test_examples["sentence2"].append(s2)
		test_examples["is_translation"].append(1)

		s1, s2 = curr_example["translation"][LANG1], random_negative_example["translation"][LANG2]
		if _is_reverse:
			s1, s2 = s2, s1

		test_examples["sentence1"].append(s1)
		test_examples["sentence2"].append(s2)
		test_examples["is_translation"].append(0)

	train_examples = pd.DataFrame(train_examples)
	dev_examples = pd.DataFrame(dev_examples)
	test_examples = pd.DataFrame(test_examples)
	print(f"{train_examples.shape[0]} train examples, "
		  f"{dev_examples.shape[0]} dev examples, "
		  f"{test_examples.shape[0]} test examples")

	train_examples.to_csv(f"id_ccmatrix_{LANG1}-{LANG2}_bitext_train_mixeddir.tsv", sep="\t", index=False)
	dev_examples.to_csv(f"id_ccmatrix_{LANG1}-{LANG2}_bitext_dev_mixeddir.tsv", sep="\t", index=False)
	test_examples.to_csv(f"id_ccmatrix_{LANG1}-{LANG2}_bitext_test_mixeddir.tsv", sep="\t", index=False)
