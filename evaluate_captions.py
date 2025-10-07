# evaluate_captions.py
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
import pandas as pd

def evaluate_metrics(predictions, references):
    # Prepare for BLEU
    gts = {str(i): [references[i]] for i in range(len(references))}
    res = {str(i): [predictions[i]] for i in range(len(predictions))}

    bleu_scorer = Bleu()
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    
    # Print all BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    print(f"\nBLEU-1 score: {bleu_score[0]:.4f}")
    print(f"BLEU-2 score: {bleu_score[1]:.4f}")
    print(f"BLEU-3 score: {bleu_score[2]:.4f}")
    print(f"BLEU-4 score: {bleu_score[3]:.4f}")

    # Prepare for CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    print(f"CIDEr score: {cider_score:.4f}")

def main():
    # Load from CSV
    df = pd.read_csv("generated_captions.csv")

    predictions = df["prediction"].tolist()
    references = df["reference"].tolist()

    print("Evaluating BLEU and CIDEr metrics...")
    evaluate_metrics(predictions, references)

if __name__ == "__main__":
    main()
