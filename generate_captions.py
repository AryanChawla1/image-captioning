# Colab cell: generate captions (inference-safe)
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput  # ✅ add this import
from tqdm import tqdm
from train_captioner import CLIP2DistilBART  # reuse the model class
import os

class Flickr8kEmbeddedTestDataset(Dataset):
    def __init__(self, parquet_path: str):
        self.data = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_emb = torch.tensor(row["image_embeds"], dtype=torch.float32)
        image_id = row.get("image_id", idx)
        return {"image_emb": image_emb, "image_id": image_id}

def inference_collate_fn(batch):
    image_embs = torch.stack([b["image_emb"] for b in batch])
    image_ids = [b.get("image_id") for b in batch]
    return {"image_emb": image_embs, "image_id": image_ids}

@torch.no_grad()
def generate_captions(model, tokenizer, dataloader, device, output_path="generated_captions.csv"):
    model.eval()
    results = []
    for batch in tqdm(dataloader, desc="Generating captions"):
        image_embs = batch["image_emb"].to(device)

        # Project CLIP embeddings into BART hidden space and add seq dim
        encoder_hidden = model.projection(image_embs).unsqueeze(1)  # [B, 1, d_model]

        # ✅ wrap it properly for BART.generate()
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        generated_ids = model.bart.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=tokenizer.bos_token_id,  # ✅ required for encoder-decoder models
            max_length=64,
            num_beams=5,
            early_stopping=True,
            do_sample=False,
            length_penalty=1.0,  # ✅ avoids NoneType error
        )


        captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        image_ids = batch["image_id"]

        for img_id, caption in zip(image_ids, captions):
            results.append({"image_id": img_id, "caption": caption})

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Captions saved to {output_path}")
    return df

# -------------------------
# Main block
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "caption_model_final"  # ensure no './' prefix
test_path = "embedded_data/flickr8k_test_embedded.parquet"

print("Loading tokenizer and trained model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = CLIP2DistilBART()
model.bart = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.projection.load_state_dict(torch.load(os.path.join(model_dir, "projection.pt"), map_location=device))
model.to(device)

print("Preparing test data...")
test_ds = Flickr8kEmbeddedTestDataset(test_path)
test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=inference_collate_fn)

print("Generating captions...")
df = generate_captions(model, tokenizer, test_dl, device)
df.head()
