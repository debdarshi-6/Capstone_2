import os
from rouge_score import rouge_scorer
import evaluate

def calculate_generative_metrics(predictions, references):
    """
    Calculates ROUGE-L and BLEU scores for generated text against reference text.
    """
    print("Calculating language generation metrics...")
    
    # 1. ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions, references)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    
    # 2. BLEU
    bleu = evaluate.load("bleu")
    # BLEU expects references as a list of lists of strings
    bleu_results = bleu.compute(predictions=predictions, references=[[r] for r in references])
    
    print("\n--- Results ---")
    print(f"Average ROUGE-L: {avg_rouge:.4f}")
    print(f"Average BLEU Score: {bleu_results['bleu']:.4f}")
    
    return {"rougeL": avg_rouge, "bleu": bleu_results['bleu']}

if __name__ == "__main__":
    # Example usage:
    preds = ["The candidate is a strong match because they possess Python, SQL, and AWS skills."]
    refs = ["Candidate matches well due to their experience in Python, SQL, and AWS cloud."]
    calculate_generative_metrics(preds, refs)
