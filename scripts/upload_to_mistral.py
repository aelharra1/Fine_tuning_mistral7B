import os
import argparse
import getpass
from mistral import Mistral  # Assurez-vous que la librairie est installée

# Définir le dossier contenant les fichiers à uploader
DATASET_DIR = "data/processed"

def upload_file(client, file_path):
    """Upload un fichier vers Mistral et retourne son ID."""
    file_name = os.path.basename(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    try:
        with open(file_path, "rb") as file_content:
            uploaded_file = client.files.upload(
                file={"file_name": file_name, "content": file_content}
            )
        print(f"Fichier uploadé : {file_name} (ID : {uploaded_file.id})")
        return uploaded_file.id
    except Exception as e:
        print(f"Erreur lors de l'upload de {file_name} : {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uploader des fichiers vers Mistral AI.")
    parser.add_argument("train_file", type=str, help="Nom du fichier d'entraînement (dans data/processed)")
    parser.add_argument("validation_file", type=str, help="Nom du fichier de validation (dans data/processed)")
    args = parser.parse_args()

    # Demander la clé API de Mistral de manière sécurisée
    api_key = getpass.getpass("Entrez votre clé API Mistral : ")

    # Initialiser le client Mistral
    client = Mistral(api_key=api_key)

    # Construire les chemins complets des fichiers
    train_path = os.path.join(DATASET_DIR, args.train_file)
    validation_path = os.path.join(DATASET_DIR, args.validation_file)

    # Uploader les fichiers
    train_file_id = upload_file(client, train_path)
    validation_file_id = upload_file(client, validation_path)

    print("\n Résumé de l'upload :")
    print(f"Train File ID      : {train_file_id}")
    print(f"Validation File ID : {validation_file_id}")