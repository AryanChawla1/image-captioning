import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from typing import Dict, Any, List

# 1. Configure data directory
COCO_ROOT = "/content/coco"
os.makedirs("data", exist_ok=True)
print(f"Saving processed files to ./data")

# 2. Load annotations
print("Loading the yerevann/coco-karpathy dataset...")
dataset: DatasetDict = load_dataset("yerevann/coco-karpathy")

# 3. Function to flatten each of the 5 captions per image to become 5 rows
def flatten_captions(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    new_rows = {
        "image_path": [],
        "caption": [],
        "imgid": [],
        "cocoid": []
    }

    # Iterate through the original batch size, determined by the length of the "filename" column
    for i in range(len(batch["filename"])):
        # Build path to local COCO image
        rel_dir = batch["filepath"][i]   # eg. train2014
        filename = batch["filename"][i]  # eg. COCO_train2014_00000012345.jpg
        abs_path = os.path.join(COCO_ROOT, rel_dir, filename)

        # Iterate over sentences list
        for sent in batch["sentences"][i]:
            new_rows["image_path"].append(abs_path)
            new_rows["caption"].append(sent)
            new_rows["imgid"].append(batch["imgid"][i])
            new_rows["cocoid"].append(batch["cocoid"][i])

    return new_rows

# 4. Process splits and save to separate parquet files
# Combine 'train' and 'restval' splits
combined_train_ds = concatenate_datasets([dataset["train"], dataset["restval"]])

# Create a new DatasetDict to hold the combined 'train' split and the other splits
combined_ds = DatasetDict({
    "train": combined_train_ds,
    "validation": dataset["validation"],
    "test": dataset["test"]
})

# Iterate through each split (train, validation, test)
for split_name, ds in combined_ds.items():
    output = f"data/coco_{split_name}.parquet"
    print(f"\nProcessing '{split_name}' split and saving to '{output}'...")

    # Apply the flattening function
    processed_ds = ds.map(
        flatten_captions,
        batched=True,
        remove_columns=["filepath", "sentids", "filename", "split", "sentences", "url"]  # Remove original columns
    )

    # Save the processed split to a Parquet file
    processed_ds.to_parquet(output)
    print(f"'{split_name}' saved successfully with {len(processed_ds)} total rows.")

print(f"\nThree separate files have been generated in the 'data' directory.")

df = pd.read_parquet("data/coco_train.parquet")
print(df.head())
