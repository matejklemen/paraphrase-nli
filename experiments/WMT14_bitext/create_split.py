import argparse
from copy import deepcopy

import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--lang_pair", default="cs-en",
					choices=["cs-en", "de-en", "fr-en", "hi-en", "ru-en"])

if __name__ == "__main__":
	args = parser.parse_args()
	LANG_PAIR = args.lang_pair
	LANG1, LANG2 = LANG_PAIR.split("-")

	data = datasets.load_dataset("wmt14", LANG_PAIR)
	train_data = data["train"]["translation"]
	dev_data = data["validation"]["translation"]
	test_data = data["test"]["translation"]

	train_examples = {
		"sentence1": [],
		"sentence2": [],
		"is_translation": []
	}
	dev_examples = deepcopy(train_examples)
	test_examples = deepcopy(train_examples)

	for _idx_ex, (_idx_neg_ex, _is_reverse) in tqdm(enumerate(zip(np.random.permutation(len(train_data)),
																  np.random.random(len(train_data)) < 0.5)),
													total=len(train_data)):
		curr_example = train_data[_idx_ex]
		random_negative_example = train_data[_idx_neg_ex]

		if _is_reverse:
			train_examples["sentence1"].append(curr_example[LANG2])
			train_examples["sentence2"].append(curr_example[LANG1])
		else:
			train_examples["sentence1"].append(curr_example[LANG1])
			train_examples["sentence2"].append(curr_example[LANG2])
		train_examples["is_translation"].append(1)

		if _is_reverse:
			train_examples["sentence1"].append(random_negative_example[LANG2])
			train_examples["sentence2"].append(curr_example[LANG1])
		else:
			train_examples["sentence1"].append(curr_example[LANG1])
			train_examples["sentence2"].append(random_negative_example[LANG2])
		train_examples["is_translation"].append(0)

	for _idx_ex, (_idx_neg_ex, _is_reverse) in tqdm(enumerate(zip(np.random.permutation(len(dev_data)),
																  np.random.random(len(dev_data)) < 0.5)),
													total=len(dev_data)):
		curr_example = dev_data[_idx_ex]
		random_negative_example = dev_data[_idx_neg_ex]

		if _is_reverse:
			dev_examples["sentence1"].append(curr_example[LANG2])
			dev_examples["sentence2"].append(curr_example[LANG1])
		else:
			dev_examples["sentence1"].append(curr_example[LANG1])
			dev_examples["sentence2"].append(curr_example[LANG2])
		dev_examples["is_translation"].append(1)

		if _is_reverse:
			dev_examples["sentence1"].append(random_negative_example[LANG2])
			dev_examples["sentence2"].append(curr_example[LANG1])
		else:
			dev_examples["sentence1"].append(curr_example[LANG1])
			dev_examples["sentence2"].append(random_negative_example[LANG2])
		dev_examples["is_translation"].append(0)

	for _idx_ex, (_idx_neg_ex, _is_reverse) in tqdm(enumerate(zip(np.random.permutation(len(test_data)),
																  np.random.random(len(test_data)) < 0.5)),
													total=len(test_data)):
		curr_example = test_data[_idx_ex]
		random_negative_example = test_data[_idx_neg_ex]

		if _is_reverse:
			test_examples["sentence1"].append(curr_example[LANG2])
			test_examples["sentence2"].append(curr_example[LANG1])
		else:
			test_examples["sentence1"].append(curr_example[LANG1])
			test_examples["sentence2"].append(curr_example[LANG2])
		test_examples["is_translation"].append(1)

		if _is_reverse:
			test_examples["sentence1"].append(random_negative_example[LANG2])
			test_examples["sentence2"].append(curr_example[LANG1])
		else:
			test_examples["sentence1"].append(curr_example[LANG1])
			test_examples["sentence2"].append(random_negative_example[LANG2])
		test_examples["is_translation"].append(0)

	train_examples = pd.DataFrame(train_examples)
	dev_examples = pd.DataFrame(dev_examples)
	test_examples = pd.DataFrame(test_examples)
	print(f"{train_examples.shape[0]} train examples, "
		  f"{dev_examples.shape[0]} dev examples, "
		  f"{test_examples.shape[0]} test examples")

	train_examples.to_csv(f"id_wmt14_{LANG_PAIR}_bitext_train_mixeddir.tsv", sep="\t", index=False)
	dev_examples.to_csv(f"id_wmt14_{LANG_PAIR}_bitext_dev_mixeddir.tsv", sep="\t", index=False)
	test_examples.to_csv(f"id_wmt14_{LANG_PAIR}_bitext_test_mixeddir.tsv", sep="\t", index=False)













