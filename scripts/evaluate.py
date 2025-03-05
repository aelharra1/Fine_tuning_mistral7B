import os
import json
import time
import argparse
import getpass
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from mistral import Mistral

# Télécharger les ressources nécessaires pour NLTK
nltk.download("punkt")

def calculate_metrics(reference, candidate):
    """Calcule BLEU, ROUGE et F1-score entre une réponse attendue et une réponse générée."""

    # BLEU score (1-gram)
    bleu_score = sentence_bleu([nltk.word_tokenize(reference)], nltk.word_tokenize(candidate), weights=(1, 0, 0, 0))

    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)

    # F1 score (approximation via la similarité des tokens)
    ref_tokens = set(nltk.word_tokenize(reference))
    cand_tokens = set(nltk.word_tokenize(candidate))

    common_tokens = ref_tokens.intersection(cand_tokens)
    precision = len(common_tokens) / len(cand_tokens) if cand_tokens else 0
    recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return bleu_score, rouge_scores, f1

def evaluate_model(api_key, test_file, fine_tuned_model):
    """Effectue l'inférence sur l'ensemble de test et calcule les métriques."""
    
    # Initialiser le client Mistral
    client = Mistral(api_key=api_key)
    
    # Charger le dataset de test
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    base_model = "open-mistral-7b"
    results = []

    bleu_scores_base, bleu_scores_finetuned = [], []
    rouge1_base, rouge2_base, rougeL_base = [], [], []
    rouge1_finetuned, rouge2_finetuned, rougeL_finetuned = [], [], []
    f1_scores_base, f1_scores_finetuned = [], []

    for i, example in enumerate(test_data):
        user_input = example["messages"][0]["content"]
        expected_output = example["messages"][1]["content"]

        # Générer les réponses des modèles
        chat_response_finetuned = client.chat.complete(model=fine_tuned_model, messages=[{"role": "user", "content": user_input}])
        chat_response_base = client.chat.complete(model=base_model, messages=[{"role": "user", "content": user_input}])

        fine_tuned_output = chat_response_finetuned.choices[0].message.content
        base_output = chat_response_base.choices[0].message.content

        # Calculer les métriques
        bleu_finetuned, rouge_finetuned, f1_finetuned = calculate_metrics(expected_output, fine_tuned_output)
        bleu_base, rouge_base, f1_base = calculate_metrics(expected_output, base_output)

        # Stocker les métriques
        bleu_scores_base.append(bleu_base)
        bleu_scores_finetuned.append(bleu_finetuned)

        rouge1_base.append(rouge_base["rouge1"].fmeasure)
        rouge2_base.append(rouge_base["rouge2"].fmeasure)
        rougeL_base.append(rouge_base["rougeL"].fmeasure)

        rouge1_finetuned.append(rouge_finetuned["rouge1"].fmeasure)
        rouge2_finetuned.append(rouge_finetuned["rouge2"].fmeasure)
        rougeL_finetuned.append(rouge_finetuned["rougeL"].fmeasure)

        f1_scores_base.append(f1_base)
        f1_scores_finetuned.append(f1_finetuned)

        # Stocker les résultats
        results.append({
            "question": user_input,
            "expected_answer": expected_output,
            "generated_answer": {
                "fine_tuned": fine_tuned_output,
                "base": base_output
            },
            "metrics": {
                "bleu": {"fine_tuned": bleu_finetuned, "base": bleu_base},
                "rouge1": {"fine_tuned": rouge_finetuned["rouge1"].fmeasure, "base": rouge_base["rouge1"].fmeasure},
                "rouge2": {"fine_tuned": rouge_finetuned["rouge2"].fmeasure, "base": rouge_base["rouge2"].fmeasure},
                "rougeL": {"fine_tuned": rouge_finetuned["rougeL"].fmeasure, "base": rouge_base["rougeL"].fmeasure},
                "f1": {"fine_tuned": f1_finetuned, "base": f1_base},
            }
        })

        # Afficher les résultats en temps réel
        print(f"🔹 {i+1}/{len(test_data)} - Question: {user_input}")
        print(f"   Expected: {expected_output}")
        print(f"   Fine-Tuned Model: {fine_tuned_output}")
        print(f"   Base Model: {base_output}\n")

        time.sleep(1)  # Pause pour éviter de surcharger l'API

    # Calcul des moyennes des scores
    avg_bleu_base = sum(bleu_scores_base) / len(bleu_scores_base)
    avg_bleu_finetuned = sum(bleu_scores_finetuned) / len(bleu_scores_finetuned)

    avg_rouge1_base = sum(rouge1_base) / len(rouge1_base)
    avg_rouge2_base = sum(rouge2_base) / len(rouge2_base)
    avg_rougeL_base = sum(rougeL_base) / len(rougeL_base)

    avg_rouge1_finetuned = sum(rouge1_finetuned) / len(rouge1_finetuned)
    avg_rouge2_finetuned = sum(rouge2_finetuned) / len(rouge2_finetuned)
    avg_rougeL_finetuned = sum(rougeL_finetuned) / len(rougeL_finetuned)

    avg_f1_base = sum(f1_scores_base) / len(f1_scores_base)
    avg_f1_finetuned = sum(f1_scores_finetuned) / len(f1_scores_finetuned)

    # Création des graphiques
    labels_bleu = ["BLEU Base", "BLEU Fine-Tuned"]
    values_bleu = [avg_bleu_base, avg_bleu_finetuned]

    labels_rouge = ["ROUGE-1 Base", "ROUGE-1 Fine-Tuned", "ROUGE-2 Base", "ROUGE-2 Fine-Tuned", "ROUGE-L Base", "ROUGE-L Fine-Tuned"]
    values_rouge = [avg_rouge1_base, avg_rouge1_finetuned, avg_rouge2_base, avg_rouge2_finetuned, avg_rougeL_base, avg_rougeL_finetuned]

    labels_f1 = ["F1 Base", "F1 Fine-Tuned"]
    values_f1 = [avg_f1_base, avg_f1_finetuned]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].bar(labels_bleu, values_bleu, color=['blue', 'red'])
    axs[0].set_title('BLEU Score Comparison')

    axs[1].bar(labels_rouge, values_rouge, color=['blue', 'red', 'blue', 'red', 'blue', 'red'])
    axs[1].set_title('ROUGE Score Comparison')

    axs[2].bar(labels_f1, values_f1, color=['blue', 'red'])
    axs[2].set_title('F1 Score Comparison')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle Mistral fine-tuné.")
    parser.add_argument("test_file", type=str, help="Fichier JSONL du test")
    parser.add_argument("fine_tuned_model", type=str, help="Nom du modèle fine-tuné")
    args = parser.parse_args()

    api_key = getpass.getpass("Entrez votre clé API Mistral : ")
    evaluate_model(api_key, args.test_file, args.fine_tuned_model)
