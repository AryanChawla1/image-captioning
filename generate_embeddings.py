import os
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoModel
from PIL import Image
from typing import Dict, Any, List

# 1. Configuration
# Input dir is from /data
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
OUTPUT_DIR = "embedded_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 2. Load models
print(f"Loading CLIP model: {CLIP_MODEL_ID}")
try:
    # Load the processor (use_fast makes it faster, suggested by hugging face)
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID, use_fast=True)
    # Load the model and move it to the selected device
    model = AutoModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
    model.eval() # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("Ensure you have 'transformers' and 'torch' installed correctly.")
    exit()

# 3. Load dataset
print(f"Loading processed Parquet files from /data")
dataset = DatasetDict({
    "train": Dataset.from_parquet("data/flickr8k_train.parquet"),
    "validation": Dataset.from_parquet("data/flickr8k_validation.parquet"),
    "test": Dataset.from_parquet("data/flickr8k_test.parquet")
})

# 4. Embedding function
@torch.no_grad()
def generate_clip_embeddings(batch: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    '''Process batch of images and captions and generate vector CLIP embeddings'''
    
    # I. Process inputs using the CLIP processor
    # The processor handles image resizing/normalization and text tokenization
    inputs = processor(
        text=batch["caption"], 
        images=batch["image"], 
        return_tensors="pt", 
        padding=True
    ).to(DEVICE)
    
    # II. Run the model inference
    outputs = model(**inputs)
    
    # III. Extract the pooled features (the final vector representations)
    # Convert tensors to numpy arrays for efficient storage in the dataset
    image_embeds = outputs.image_embeds.to("cpu").numpy()
    text_embeds = outputs.text_embeds.to("cpu").numpy()
    
    # IV. Return results
    # The map function expects a dictionary of new columns
    return {
        "image_embeds": image_embeds,
        "text_embeds": text_embeds
    }

# 5. Generate embeddings and save results
# Iterate over each split
for split_name, ds in dataset.items():
    output_filepath = os.path.join(OUTPUT_DIR, f"flickr8k_{split_name}_embedded.parquet")
    print(f"\nGenerating CLIP embeddings for '{split_name}' split")
    
    # Apply the embedding function
    embedded_ds = ds.map(
        generate_clip_embeddings, 
        batched=True,
        batch_size=32, # Can be increased for faster processing
    )
    
    # Save the processed dataset to a new Parquet file
    # Original image/caption plus the two new embedding columns
    embedded_ds.to_parquet(output_filepath)
    
    print(f"'{split_name}' embedded and saved successfully to '{output_filepath}'.")

print("\nEmbedding Generation Complete")
print(f"The final embedded datasets are ready in the '{OUTPUT_DIR}' directory.")
print("The new files include 'image_embeds' and 'text_embeds' columns, which are the CLIP vectors.")