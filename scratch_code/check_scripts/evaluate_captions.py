import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

# Load generated captions
gen_df = pd.read_csv("generated_captions.csv")

# Load ground-truth captions
gt_df = pd.read_parquet("embedded_data/coco_test_embedded.parquet")

# Prepare predictions dict
predictions = {}
for _, row in gen_df.iterrows():
    img_id = int(row["image_id"])
    predictions[img_id] = [str(row["caption"])]

# Prepare references dict
references = {}
for _, row in gt_df.iterrows():
    img_id = int(row["imgid"])
    caption = str(row["caption"])
    references.setdefault(img_id, []).append(caption)

# Filter to matching image IDs only
common_ids = set(predictions.keys()) & set(references.keys())
predictions = {k: predictions[k] for k in common_ids}
references = {k: references[k] for k in common_ids}

# Initialize scorers
bleu_scorer = Bleu(4)  # BLEU-1 to BLEU-4
cider_scorer = Cider()

# Compute scores
bleu_scores, _ = bleu_scorer.compute_score(references, predictions)
cider_score, _ = cider_scorer.compute_score(references, predictions)

print(f"BLEU-4 score: {bleu_scores[3]:.4f}")
print(f"CIDEr score: {cider_score:.4f}")
