import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm


ALL_LANGUAGES = ['af', 'ar', 'az', 'be', 'ber', 'bg', 'bn', 'br', 'ca', 'cbk', 'cmn', 'cs', 'da', 'de', 'el', 'en',
				 'eo', 'es', 'et', 'eu', 'fi', 'fr', 'gl', 'gos', 'he', 'hi', 'hr', 'hu', 'hy', 'ia', 'id', 'ie', 'io',
				 'is', 'it', 'ja', 'jbo', 'kab', 'ko', 'kw', 'la', 'lfn', 'lt', 'mk', 'mr', 'nb', 'nds', 'nl', 'orv',
				 'ota', 'pes', 'pl', 'pt', 'rn', 'ro', 'ru', 'sl', 'sr', 'sv', 'tk', 'tl', 'tlh', 'toki', 'tr', 'tt',
				 'ug', 'uk', 'ur', 'vi', 'vo', 'war', 'wuu', 'yue']


if __name__ == "__main__":
	LANGS_TO_CONSIDER = ALL_LANGUAGES
	train_pairs = []
	dev_pairs = []
	test_pairs = []

	for curr_lang in LANGS_TO_CONSIDER:
		data = datasets.load_dataset("tapaco", curr_lang)["train"]
		print(f"Loaded dataset '{curr_lang}'!")

		# Group sentences by paraphrase set ID
		set_to_sentences = {}
		for ex in tqdm(data):
			if ex["paraphrase_set_id"] not in set_to_sentences:
				set_to_sentences[ex["paraphrase_set_id"]] = [ex["paraphrase"]]
			else:
				set_to_sentences[ex["paraphrase_set_id"]].append(ex["paraphrase"])

		paraphrases = list(set_to_sentences.items())  # list of tuples (ID_set, list_of_paras)
		print(f"{len(data)} examples in {len(set_to_sentences)} paraphrase sets")

		curr_pairs = {"sentence1": [], "sentence2": [], "is_paraphrase": [], "language": []}
		indices = np.arange(len(paraphrases))
		for idx_set, (set_id, paras) in enumerate(paraphrases):

			# 1. Create paraphrases by pairing adjacent sentences inside paraphrase set
			num_para_pairs = len(paras) // 2
			for _idx_pair in range(num_para_pairs):
				curr_pairs["sentence1"].append(paras[_idx_pair * 2])
				curr_pairs["sentence2"].append(paras[_idx_pair * 2 + 1])
				curr_pairs["is_paraphrase"].append(1)
				curr_pairs["language"].append(curr_lang)

			# 2. Create non-paraphrases by pairing first sentence in paraphrases with a random sentence from
			#  a different paraphrase set
			valid_mask = np.ones(len(paraphrases), dtype=bool)
			valid_mask[idx_set] = False
			selected_nonpara_set = np.random.choice(indices[valid_mask], size=num_para_pairs)

			for _idx_pair, _idx_set in enumerate(selected_nonpara_set):
				available_sents = paraphrases[_idx_set][1]
				nonpara_sent = available_sents[np.random.choice(np.arange(len(available_sents)), size=1)[0]]

				curr_pairs["sentence1"].append(paras[_idx_pair * 2])
				curr_pairs["sentence2"].append(nonpara_sent)
				curr_pairs["is_paraphrase"].append(0)
				curr_pairs["language"].append(curr_lang)

		curr_pairs = pd.DataFrame(curr_pairs)
		num_pairs = curr_pairs.shape[0]
		print(f"{num_pairs} paraphrase identification examples created for '{curr_lang}'")
		print(curr_pairs["is_paraphrase"].value_counts())
		print("")

		shuffled_indices = np.arange(num_pairs)

		# 70/15/15% split per language
		train_pairs.append(curr_pairs.iloc[shuffled_indices[:int(0.7 * num_pairs)]])
		dev_pairs.append(curr_pairs.iloc[shuffled_indices[int(0.7 * num_pairs): int(0.85 * num_pairs)]])
		test_pairs.append(curr_pairs.iloc[shuffled_indices[int(0.85 * num_pairs):]])

	train_pairs = pd.concat(train_pairs).reset_index(drop=True)
	dev_pairs = pd.concat(dev_pairs).reset_index(drop=True)
	test_pairs = pd.concat(test_pairs).reset_index(drop=True)
	print(f"{LANGS_TO_CONSIDER}:\n"
		  f"{train_pairs.shape[0]} train, "
		  f"{dev_pairs.shape[0]} dev, "
		  f"{test_pairs.shape[0]} test examples")

	if len(LANGS_TO_CONSIDER) < 10:
		lang_group_name = "_".join(LANGS_TO_CONSIDER)
	elif len(LANGS_TO_CONSIDER) == len(ALL_LANGUAGES):
		lang_group_name = f"all_languages"
	else:
		lang_group_name = f"{len(LANGS_TO_CONSIDER)}languages"

	# Combine only paraphrases, which will get verified in the reverse direction
	all_paras = pd.concat((train_pairs, dev_pairs, test_pairs), axis=0)
	total_examples = all_paras.shape[0]
	all_paras = all_paras.loc[all_paras["is_paraphrase"] == 1]
	print(f"{all_paras.shape[0]}/{total_examples} paraphrases")
	all_paras.to_csv(f"id_tapaco_{lang_group_name}_onlyparas.tsv", sep="\t", index=False)

	dev_pairs = dev_pairs.sample(n=20_000)
	test_pairs = test_pairs.sample(n=20_000)
	print(dev_pairs["language"].value_counts())
	print(test_pairs["language"].value_counts())

	train_pairs.to_csv(f"id_tapaco_{lang_group_name}_train.tsv", sep="\t", index=False)
	dev_pairs.to_csv(f"id_tapaco_{lang_group_name}_dev.tsv", sep="\t", index=False)
	test_pairs.to_csv(f"id_tapaco_{lang_group_name}_test.tsv", sep="\t", index=False)









