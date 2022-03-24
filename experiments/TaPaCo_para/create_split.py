import datasets
import numpy as np
import pandas as pd

if __name__ == "__main__":
	LANGS_TO_CONSIDER = ["en"]
	train_pairs = []
	dev_pairs = []
	test_pairs = []

	for curr_lang in LANGS_TO_CONSIDER:
		data = datasets.load_dataset("tapaco", curr_lang)["train"]
		pairs = {
			"sentence1": [], "sentence2": [], "is_paraphrase": [], "language": []
		}

		# Group sentences by paraphrase set ID
		set_to_sentences = {}
		for ex in data:
			if ex["paraphrase_set_id"] not in set_to_sentences:
				set_to_sentences[ex["paraphrase_set_id"]] = [ex["paraphrase"]]
			else:
				set_to_sentences[ex["paraphrase_set_id"]].append(ex["paraphrase"])

		paraphrases = list(set_to_sentences.items())  # list of tuples (ID_set, list_of_paras)

		indices = np.arange(len(paraphrases))
		for idx_set, (set_id, paras) in enumerate(paraphrases):

			# 1. Create paraphrases by pairing adjacent sentences inside paraphrase set
			num_para_pairs = len(paras) // 2
			for _idx_pair in range(num_para_pairs):
				pairs["sentence1"].append(paras[_idx_pair * 2])
				pairs["sentence2"].append(paras[_idx_pair * 2 + 1])
				pairs["is_paraphrase"].append(1)
				pairs["language"].append(curr_lang)

			# 2. Create non-paraphrases by pairing first sentence in paraphrases with a random sentence from
			#  a different paraphrase set
			valid_mask = np.ones(len(paraphrases), dtype=bool)
			valid_mask[idx_set] = False
			selected_nonpara_set = np.random.choice(indices[valid_mask], size=num_para_pairs)

			for _idx_pair, _idx_set in enumerate(selected_nonpara_set):
				available_sents = paraphrases[_idx_set][1]
				nonpara_sent = available_sents[np.random.choice(np.arange(len(available_sents)), size=1)[0]]

				pairs["sentence1"].append(paras[_idx_pair * 2])
				pairs["sentence2"].append(nonpara_sent)
				pairs["is_paraphrase"].append(0)
				pairs["language"].append(curr_lang)

		print(f"{len(data)} examples in {len(set_to_sentences)} paraphrase sets")
		pairs = pd.DataFrame(pairs)
		num_pairs = pairs.shape[0]
		shuffled_indices = np.arange(num_pairs)

		# 70/15/15% split per language
		train_pairs.append(pairs.iloc[shuffled_indices[:int(0.7 * num_pairs)]])
		dev_pairs.append(pairs.iloc[shuffled_indices[int(0.7 * num_pairs): int(0.85 * num_pairs)]])
		test_pairs.append(pairs.iloc[shuffled_indices[int(0.85 * num_pairs):]])

	train_pairs = pd.concat(train_pairs).reset_index(drop=True)
	dev_pairs = pd.concat(dev_pairs).reset_index(drop=True)
	test_pairs = pd.concat(test_pairs).reset_index(drop=True)
	print(f"{LANGS_TO_CONSIDER}:\n"
		  f"{train_pairs.shape[0]} train, "
		  f"{dev_pairs.shape[0]} dev, "
		  f"{test_pairs.shape[0]} test examples")

	train_pairs.to_csv("id_tapaco_{}_train.tsv".format("_".join(LANGS_TO_CONSIDER)), sep="\t", index=False)
	dev_pairs.to_csv("id_tapaco_{}_dev.tsv".format("_".join(LANGS_TO_CONSIDER)), sep="\t", index=False)
	test_pairs.to_csv("id_tapaco_{}_test.tsv".format("_".join(LANGS_TO_CONSIDER)), sep="\t", index=False)








