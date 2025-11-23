import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from transformers import (
   AutoTokenizer,
   AutoModelForSeq2SeqLM,
   Trainer,
   TrainingArguments,
)
from typing import Dict

# 1. Dataset class
class COCOEmbeddedDataset(Dataset):
   def __init__(self, parquet_path: str, tokenizer, max_length: int = 64):
      self.data = pd.read_parquet(parquet_path)
      self.tokenizer = tokenizer
      self.max_length = max_length

   def __len__(self):
      return len(self.data)

   def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
      row = self.data.iloc[idx]

      # Convert embedding to tensor
      image_emb = torch.tensor(row["image_embeds"], dtype=torch.float32)

      # Tokenize caption text
      tokenized = self.tokenizer(
         row["caption"],
         truncation=True,
         padding="max_length",
         max_length=self.max_length,
         return_tensors="pt",
      )

      return {
         "image_emb": image_emb,
         "labels": tokenized["input_ids"].squeeze(),
      }

# 2. Model definition
class CLIP2DistilBART(nn.Module):
   def __init__(self, clip_emb_dim=512, bart_model_name="sshleifer/distilbart-cnn-12-6"):
      super().__init__()
      self.bart = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name)
      self.projection = nn.Linear(clip_emb_dim, self.bart.config.d_model)

   def forward(self, image_emb, labels=None):
      # Map 512-dim CLIP vector to DistilBART encoder hidden size (e.g., 768)
      encoder_hidden = self.projection(image_emb).unsqueeze(1)  # [batch, 1, hidden_dim]
      outputs = self.bart(
         encoder_outputs=(encoder_hidden,),
         labels=labels,
      )
      return outputs

# 3. Collate function
def collate_fn(batch):
   image_embs = torch.stack([b["image_emb"] for b in batch])
   labels = torch.stack([b["labels"] for b in batch])
   return {
      "image_emb": image_embs,
      "labels": labels,
   }

# 4. Training entry point
def main():
   model_name = "sshleifer/distilbart-cnn-12-6"
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   print("Loading embedded COCO data...")
   train_ds = COCOEmbeddedDataset("embedded_data/coco_train_embedded.parquet", tokenizer)
   val_ds = COCOEmbeddedDataset("embedded_data/coco_validation_embedded.parquet", tokenizer)

   print("Initializing CLIP2DistilBART model...")
   model = CLIP2DistilBART(clip_emb_dim=512, bart_model_name=model_name)

   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using device: {device}")
   model.to(device)

   # Training hyperparameters
   args = TrainingArguments(
      output_dir="./caption_model_final",
      per_device_train_batch_size=16,
      num_train_epochs=3,
      learning_rate=5e-5,
      logging_steps=100,
      save_strategy="no",
      do_eval=False,
      fp16=torch.cuda.is_available(),
      report_to=[],
   )

   trainer = Trainer(
      model=model,
      args=args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      data_collator=collate_fn,
   )

   print("Starting training...")
   trainer.train()

   print("Saving model...")
   model.bart.save_pretrained("caption_model_final", safe_serialization=False)
   tokenizer.save_pretrained("caption_model_final")
   torch.save(model.projection.state_dict(), "caption_model_final/projection.pt")
   print("Training complete. Model saved to caption_model_final")

if __name__ == "__main__":
   main()
