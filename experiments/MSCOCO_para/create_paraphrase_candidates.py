# Script to extract alleged paraphrases, i.e. captions of same image;
# This is NOT the script to extract train/dev/test sets for training a paraphrase identification model!

import json
import os

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    DATA_DIR = "/home/matej/Documents/data/MSCOCO"

    train_path = os.path.join(DATA_DIR, "captions_train2014.json")
    dev_path = os.path.join(DATA_DIR, "captions_val2014.json")

    data = {
        "seq1": [], "seq2": [], "is_paraphrase": [],
        "im_id": [], "cap1_id": [], "cap2_id": []
    }

    for dataset_name, data_path in [("train", train_path), ("dev", dev_path)]:
        print(f"Processing dataset '{dataset_name}'")
        with open(data_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        image_ids = []
        image_to_captions = {}
        captions = {}

        # Group captions by image IDs
        for curr_annotation in content["annotations"]:
            im_id = curr_annotation["image_id"]
            cap_id = curr_annotation["id"]
            caption = curr_annotation["caption"]

            image_ids.append(im_id)
            existing_captions = image_to_captions.get(im_id, [])
            existing_captions.append(cap_id)
            image_to_captions[im_id] = existing_captions
            captions[cap_id] = caption.strip()

        # Pair first caption for image with all others, e.g., for 5 captions -> (1, 2), (1, 3), (1, 4), (1, 5)
        for curr_im_id, curr_captions in tqdm(image_to_captions.items(), total=len(image_to_captions)):
            first_cap_id = curr_captions[0]

            for other_cap_id in curr_captions[1:]:
                data["seq1"].append(captions[first_cap_id])
                data["seq2"].append(captions[other_cap_id])
                data["is_paraphrase"].append(1)
                data["im_id"].append(curr_im_id)
                data["cap1_id"].append(first_cap_id)
                data["cap2_id"].append(other_cap_id)

    df = pd.DataFrame(data)
    print(f"Writing {df.shape[0]} paraphrase candidates")
    df.to_csv("paraphrase_candidates.tsv", sep="\t", index=False)















