# generate_captions.py
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from train_captioner import CLIP2DistilBART
import os

# Dataset class for test set
class Flickr8kEmbeddedTestDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=64):
        self.data = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_emb = torch.tensor(row["image_embeds"], dtype=torch.float32)
        caption = row["caption"]
        return {
            "image_emb": image_emb,
            "caption": caption
        }

@torch.no_grad()
def generate_captions(model, dataset, tokenizer, device, batch_size=32):
    model.eval()
    model.to(device)

    predictions = []
    references = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        image_embs = torch.stack([item['image_emb'] for item in batch]).to(device)
        ground_truths = [item['caption'] for item in batch]

        encoder_hidden = model.projection(image_embs).unsqueeze(1)
        generated_ids = model.bart.generate(
            inputs_embeds=encoder_hidden,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded_preds)
        references.extend(ground_truths)

    return predictions, references

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading test dataset...")
    test_dataset = Flickr8kEmbeddedTestDataset(
        "embedded_data/flickr8k_test_embedded.parquet", tokenizer
    )

    print("Loading trained model...")
    model = CLIP2DistilBART()
    model.bart.from_pretrained("caption_model_final")
    model.projection.load_state_dict(torch.load("caption_model_final/projection.pt"))

    print("Generating captions...")
    predictions, references = generate_captions(model, test_dataset, tokenizer, device)

    # Save to CSV
    output_df = pd.DataFrame({
        "prediction": predictions,
        "reference": references
    })
    output_path = "generated_captions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved generated captions to {output_path}")

if __name__ == "__main__":
    main()
