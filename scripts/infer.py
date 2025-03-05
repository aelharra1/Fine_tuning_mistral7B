from mistral import Mistral
import getpass

def infer(api_key, prompt):
    """Crée et lance une inférence en utilisant l'API Mistral."""

    # Initialiser le client Mistral
    client = Mistral(api_key=api_key)
    
    # Lister les jobs (finetuning)
    jobs = client.fine_tuning.jobs.list()

    # Dernier modèle fine-tuné
    job_id = jobs.data[0].id

    # Choisir le job correspondant au modèle souhaité
    retrieved_jobs = client.fine_tuning.jobs.get(job_id=job_id)

    # Charger le modèle fine-tuné
    fine_tuned_model = retrieved_jobs.fine_tuned_model

    # Tester le modèle avec le prompt de l'utilisateur
    chat_response = client.chat.complete(
        model=fine_tuned_model,
        messages=[{"role": 'user', "content": prompt}]
    )

    # Afficher la réponse
    print("Réponse du modèle:", chat_response.choices[0].message.content)

if __name__ == "__main__":
    # Demander la clé API et la question à l'utilisateur
    api_key = getpass.getpass("Entrez votre clé API Mistral : ")
    prompt = input("Entrez votre prompt : ")

    # Lancer l'inférence
    infer(api_key, prompt)
