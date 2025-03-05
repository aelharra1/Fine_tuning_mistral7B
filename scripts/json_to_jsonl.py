# ATTENTION : Ce script suppose que les données sont organisées en "question" et "answer"

import os
import json
import argparse

# Définir les dossiers d'entrée et de sortie
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/converted"

# Vérifier et créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_json_to_jsonl(input_filename):
    """Convertit un fichier JSON en JSONL et le sauvegarde dans le dossier de sortie."""
    
    input_path = os.path.join(INPUT_DIR, input_filename)
    output_filename = os.path.splitext(input_filename)[0] + "_formatted.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Vérifier si le fichier existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")

    # Charger le fichier JSON
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data_f = json.load(f)
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier JSON : {e}")

    # Convertir les données en JSONL
    jsonl_data = [
        {
            "messages": [
                {"role": "user", "content": entry["question"]},
                {"role": "assistant", "content": entry["answer"]}
            ]
        }
        for entry in data_f
    ]

    # Sauvegarder en JSONL
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry) + '\n')

    print(f"Fichier JSONL généré : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir un fichier JSON en JSONL.")
    parser.add_argument("filename", type=str, help="Nom du fichier JSON à convertir (doit être dans data/raw)")
    args = parser.parse_args()

    convert_json_to_jsonl(args.filename)
