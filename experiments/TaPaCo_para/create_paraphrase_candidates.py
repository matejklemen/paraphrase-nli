import datasets
import numpy as np
import pandas as pd

if __name__ == "__main__":
	LANGS_TO_CONSIDER = ["en"]
	pairs = []

	# Add a dummy non-paraphrase as the data loader is for paraphrase identification, a BINARY task, so it expects
	# two unique labels
	pairs.append(
		pd.DataFrame({"sentence1": ["Dummy s1"], "sentence2": ["Dummy s2"], "is_paraphrase": [0], "language": ["xy"]})
	)

	for curr_lang in LANGS_TO_CONSIDER:
		data = datasets.load_dataset("tapaco", curr_lang)["train"]
		curr_pairs = {
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
				curr_pairs["sentence1"].append(paras[_idx_pair * 2])
				curr_pairs["sentence2"].append(paras[_idx_pair * 2 + 1])
				curr_pairs["is_paraphrase"].append(1)
				curr_pairs["language"].append(curr_lang)

		print(f"{len(data)} examples in {len(set_to_sentences)} paraphrase sets")
		pairs.append(pd.DataFrame(curr_pairs))

	pairs = pd.concat(pairs).reset_index(drop=True)
	print(f"{LANGS_TO_CONSIDER}: {pairs.shape[0]} candidates")

	pairs.to_csv("tapaco_{}_paraphrase_candidates.tsv".format("_".join(LANGS_TO_CONSIDER)), sep="\t", index=False)








