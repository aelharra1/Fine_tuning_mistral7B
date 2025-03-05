import os
import time
import argparse
import getpass
import wandb
from mistral import Mistral

# Définir les hyperparamètres par défaut
DEFAULT_HYPERPARAMS = {
    "training_steps": 10,
    "learning_rate": 0.0001,
    "weight_decay": 0.1,
    "warmup_fraction": 0.05,
}

def train_mistral(api_key, wandb_key, train_file_id, validation_file_id, hyperparams):
    """Crée et lance un job de fine-tuning sur Mistral AI tout en loggant avec WandB."""

    # Initialiser le client Mistral
    client = Mistral(api_key=api_key)

    # Créer un job de fine-tuning
    created_job = client.fine_tuning.jobs.create(
        model="open-mistral-7b",
        training_files=[{"file_id": train_file_id, "weight": 1}],
        validation_files=[validation_file_id],
        hyperparameters=hyperparams,
        auto_start=False  # Démarrer manuellement après validation
    )

    print(f" Job créé avec succès : {created_job.id}")

    # Lancer le job de fine-tuning
    client.fine_tuning.jobs.start(job_id=created_job.id)
    print(f" Entraînement démarré : {created_job.id}")

    # Initialiser WandB
    wandb.login(key=wandb_key)
    wandb.init(project="mistral-finetuning", name="Medical")

    # Stocker les étapes déjà enregistrées pour éviter les doublons
    logged_steps = set()

    while True:
        job_status = client.fine_tuning.jobs.get(job_id=created_job.id)

        # Vérifier s'il y a des checkpoints
        if job_status.checkpoints:
            checkpoint = job_status.checkpoints[0]
            step = checkpoint.step_number

            if step not in logged_steps:
                train_loss = checkpoint.metrics.train_loss
                valid_loss = checkpoint.metrics.valid_loss
                accuracy = checkpoint.metrics.valid_mean_token_accuracy

                # Envoyer les métriques à WandB
                wandb.log({
                    "step": step,
                    "training_loss": train_loss,
                    "validation_loss": valid_loss,
                    "validation_accuracy": accuracy
                })

                print(f"Step {step} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Val Acc: {accuracy:.4f}")

                logged_steps.add(step)

        else:
            print("Aucun checkpoint disponible...")

        # Vérifier si le job est terminé
        if job_status.status in ["FAILED_VALIDATION", "STOPPED", "SUCCESS"]:
            break

        time.sleep(1)  # Vérifier toutes les secondes

    # Fin du run WandB
    wandb.finish()
    print("Entraînement terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning d'un modèle Mistral avec suivi WandB.")
    parser.add_argument("train_file_id", type=str, help="ID du fichier d'entraînement sur Mistral")
    parser.add_argument("validation_file_id", type=str, help="ID du fichier de validation sur Mistral")
    parser.add_argument("--training_steps", type=int, default=DEFAULT_HYPERPARAMS["training_steps"], help="Nombre d'étapes d'entraînement")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_HYPERPARAMS["learning_rate"], help="Taux d'apprentissage")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_HYPERPARAMS["weight_decay"], help="Pondération du déclin du poids")
    parser.add_argument("--warmup_fraction", type=float, default=DEFAULT_HYPERPARAMS["warmup_fraction"], help="Fraction de l'échauffement")

    args = parser.parse_args()

    # Demander les clés API à l'utilisateur
    api_key = getpass.getpass("Entrez votre clé API Mistral : ")
    wandb_key = getpass.getpass("Entrez votre clé API WandB : ")

    # Récupérer les hyperparamètres
    hyperparams = {
        "training_steps": args.training_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction,
    }

    # Lancer l'entraînement
    train_mistral(api_key, wandb_key, args.train_file_id, args.validation_file_id, hyperparams)
