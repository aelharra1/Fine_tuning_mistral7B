import os
import json
import random
import argparse

# Définir les dossiers d'entrée et de sortie
INPUT_DIR = "data/converted"
OUTPUT_DIR = "data/processed"

# Vérifier et créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_dataset(input_filename, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    """Divise un fichier JSONL en ensembles d'entraînement, validation et test."""
    
    input_path = os.path.join(INPUT_DIR, input_filename)
    base_filename = os.path.splitext(input_filename)[0]  # Récupérer le nom sans extension

    # Vérifier si le fichier existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")

    # Charger le fichier JSONL
    dataset = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))  # Convertir chaque ligne JSON en dictionnaire

    # Mélanger les données pour garantir une distribution aléatoire
    random.shuffle(dataset)

    # Vérifier que la somme des ratios est bien égale à 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio != 1:
        raise ValueError(f"Les ratios doivent totaliser 1 (actuellement : {total_ratio})")

    # Calcul des tailles des ensembles
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Division du dataset
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]

    # Fonction pour enregistrer un dataset en JSONL
    def save_jsonl(data, output_filename):
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f" Fichier créé : {output_path} ({len(data)} échantillons)")

    # Sauvegarde des fichiers
    save_jsonl(train_data, f"{base_filename}_train.jsonl")
    save_jsonl(val_data, f"{base_filename}_validation.jsonl")
    save_jsonl(test_data, f"{base_filename}_test.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diviser un fichier JSONL en ensembles train/val/test.")
    parser.add_argument("filename", type=str, help="Nom du fichier JSONL à diviser (doit être dans data/converted)")
    args = parser.parse_args()

    split_dataset(args.filename)
