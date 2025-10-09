import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

# Load generated captions
gen_df = pd.read_csv("generated_captions.csv")

# Load ground-truth captions
gt_df = pd.read_parquet("embedded_data/flickr8k_test_embedded.parquet")

# Prepare predictions dict: {image_id: generated_caption}
predictions = {}
for i, row in gen_df.iterrows():
    predictions[i] = [str(row["caption"])]

# Prepare references dict: {image_id: [ref1, ref2, ...]}
references = {}
for i, row in gt_df.iterrows():
    image_id = i // 5  # Group every 5 captions
    references.setdefault(image_id, []).append(str(row["caption"]))

# Initialize scorers
bleu_scorer = Bleu(4)  # BLEU-1 to BLEU-4
cider_scorer = Cider()

# Compute scores
bleu_scores, _ = bleu_scorer.compute_score(references, predictions)
cider_score, _ = cider_scorer.compute_score(references, predictions)

print(f"BLEU-4 score: {bleu_scores[3]:.4f}")
print(f"CIDEr score: {cider_score:.4f}")
