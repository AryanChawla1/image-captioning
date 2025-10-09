import pandas as pd
from datasets import load_dataset, DatasetDict
from typing import Dict, Any, List
import os

# 1. Configure data directory
os.makedirs("data", exist_ok=True)
print(f"Saving processed files to /image-captioning/data")

# 2. Load dataset
print("Loading the jxie/flickr8k dataset...")
# Returns a DatasetDict with "train", "validation", and "test" splits
dataset: DatasetDict = load_dataset("jxie/flickr8k")

# 3. Function to flatten each of the 5 captions per image to become 5 rows
def flatten_captions(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    '''Takes a batch from Hugging Face Dataset and flattens it'''
    
    new_rows = {
        "image": [],
        "caption": [],
    }
    
    # Iterate through the original batch size, determined by the length of the "image" column
    for i in range(len(batch["image"])): 
        img = batch["image"][i]
        
        # Iterate over the five caption columns
        for j in range(5):
            caption_key = f"caption_{j}"
            
            new_rows["image"].append(img)
            new_rows["caption"].append(batch[caption_key][i])
            
    return new_rows

# 4. Process splits and save to separate parquet files
# Iterate through each original split (train, validation, test)
for split_name, ds in dataset.items():
    output = f"data/flickr8k_{split_name}.parquet"
    
    print(f"\nProcessing '{split_name}' split and saving to '{output}'...")
    
    # Apply the flattening function
    processed_ds = ds.map(
        flatten_captions, 
        batched=True,
        remove_columns=[f"caption_{i}" for i in range(5)],
    )

    # Save the processed split to a Parquet file
    processed_ds.to_parquet(output)
    
    print(f"'{split_name}' saved successfully with {len(processed_ds)} total rows.")

print(f"\nThree separate files have been generated in the 'data' directory.")
