# ATTENTION : Ce script suppose que les noms des colonnes du fichier csv sont "input" et "output"

import os
import json
import pandas as pd
import argparse

# Définir les dossiers d'entrée et de sortie
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/converted"

# Vérifier et créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_csv_to_jsonl(input_filename):
    """Convertit un fichier CSV en JSONL et le sauvegarde dans le dossier de sortie."""
    
    input_path = os.path.join(INPUT_DIR, input_filename)
    output_filename = os.path.splitext(input_filename)[0] + "_formatted.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Vérifier si le fichier existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")

    # Charger le fichier CSV
    try:
        data_f = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier CSV : {e}")

    # Vérifier si les colonnes nécessaires existent
    required_columns = {"input", "output"}
    if not required_columns.issubset(data_f.columns):
        raise ValueError(f"Le fichier CSV doit contenir les colonnes suivantes : {required_columns}")

    # Convertir les données en JSONL formaté
    jsonl_data = [
        {
            "messages": [
                {"role": "user", "content": row["input"]},
                {"role": "assistant", "content": row["output"]}
            ]
        }
        for _, row in data_f.iterrows()
    ]

    # Sauvegarder en JSONL
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry) + '\n')

    print(f"Fichier JSONL généré : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir un fichier CSV en JSONL.")
    parser.add_argument("filename", type=str, help="Nom du fichier CSV à convertir (doit être dans data/raw)")
    args = parser.parse_args()

    convert_csv_to_jsonl(args.filename)
