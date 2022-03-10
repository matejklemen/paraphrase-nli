import numpy as np


def mask_long_enough(col1_vals, col2_vals, min_chars=5):
	assert np.sum(mask_not_na(col1_vals, col2_vals)) == col1_vals.shape[0]
	return np.logical_not(np.logical_or(col1_vals.apply(lambda _s: len(_s)) < min_chars,
										col2_vals.apply(lambda _s: len(_s)) < min_chars))


def mask_not_na(col1_vals, col2_vals):
	return np.logical_not(np.logical_or(col1_vals.isna(), col2_vals.isna()))


def inds_unique(col1_vals, col2_vals):
	existing_pairs = set()
	uniq_indices = []
	for _i in range(col1_vals.shape[0]):
		curr_pair = (col1_vals.iloc[_i], col2_vals.iloc[_i])

		if curr_pair not in existing_pairs:
			uniq_indices.append(_i)
			existing_pairs.add(curr_pair)

	return uniq_indices
