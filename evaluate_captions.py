import pandas as pd
import evaluate

# Load generated captions
gen_df = pd.read_csv("generated_captions.csv")  # columns: image_id, caption

# Load ground-truth captions
gt_df = pd.read_parquet("embedded_data/flickr8k_test_embedded.parquet")  # columns: caption, ...

# Ensure lengths match
assert len(gen_df) == len(gt_df), "❌ Length mismatch between predictions and ground truth!"

# Prepare predictions and references
predictions = gen_df["caption"].astype(str).tolist()
final_predictions = []
for i in range(len(predictions)):
  if i % 5 == 0:
    final_predictions.append(predictions[i])
references  = gt_df["caption"].astype(str).tolist()  # single reference per prediction
final_references = []
temp = []
for i in range(len(references)):
  temp.append(references[i])
  if len(temp) == 5:
    final_references.append(temp)
    temp = []

# Load BLEU metric
bleu = evaluate.load("bleu")
cider = evaluate.load("sunhill/cider")
bleu_score = bleu.compute(predictions=final_predictions, references=final_references)
cider_score = cider.compute(predictions=final_predictions, references=final_references)

print(f"✅ Average BLEU score: {bleu_score['bleu']:.4f}")
print(f"✅ Average CIDEr score: {cider_score['cider_score']:.4f}")
