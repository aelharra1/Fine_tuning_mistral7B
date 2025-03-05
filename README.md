# Fine-tuning Mistral 7B

Ce projet permet de fine-tuner le modèle **Mistral 7B** sur un dataset json ou csv. Il comprend des scripts pour la conversion et préparation des données, l'entraînement, l'inférence et l'évaluation du modèle.

---

##  Notes
- Ce projet utilise **Weights & Biases (W&B)** pour le suivi des performances.
- Vous pouvez modifier les hyperparamètres d'entraînement dans `train.py`.
- Le notebook `projet_complet.ipynb` permet d'exécuter l'ensemble du projet sous Google Colab.
- Attention les scripts de conversion supposent que les données ont une certaine organisation sinon vous pouvez modifier directement les scripts

---

##  Prérequis

Avant de commencer, assurez-vous d'avoir :
- Une **clé API Mistral** avec des crédits disponibles
- Un **compte Weights & Biases (W&B)** pour le suivi de l'entraînement

---

##  Structure du projet
```
/MonProjet-FineTuning
│── /data
│   ├── /raw           # Dossier contenant les datasets bruts
│   ├── /converted     # Dossier contenant les datasets convertis
│   ├── /processed     # Dossier contenant train.jsonl, validation.jsonl et test.jsonl
│── /scripts
│   ├── json_to_jsonl.py  # Conversion JSON vers JSONL et au format attendu par mistral
│   ├── csv_to_jsonl.py   # Conversion CSV vers JSONL et au format attendu par mistral
│   ├── train_test_val.py     # Génération des ensembles train/test/val
│   ├── upload_data.py    # Upload des datasets sur Mistral
│   ├── train.py          # Fine-tuning du modèle
│   ├── infer.py          # Inférence sur de nouvelles données
│   ├── evaluate.py       # Calcul des métriques d’évaluation
│── /notebooks
│   ├── Fine_tuning_mistral7B.ipynb  # Notebook Google Colab complet
│── requirements.txt  # Liste des dépendances
│── README.md  # Documentation du projet
```

---

##  Installation
### 1. Cloner le repo
```bash
git clone https://github.com/aelharra1/Fine_tuning_mistral7B.git
cd Fine_tuning_mistral7B
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

##  Préparation des données
### 1. Ajouter votre dataset
Placez votre dataset brut dans `data/raw/` au format **CSV** ou **JSON**.

### 2. Convertir le dataset au format JSONL
Pour le csv, les noms des colonnes doivent être 'input' et 'output' et pour le json il doit y avoir 'question' et 'answer'

Si votre dataset est en JSON, utilisez :
```bash
python scripts/json_to_jsonl.py votre_fichier.json
```
Si votre dataset est en CSV, utilisez :
```bash
python scripts/csv_to_jsonl.py votre_fichier.csv
```
La conversion permet d'obtenir des fichiers au format demandé par Mistral pour le fine-tuning.
Cela générera `votre_fichier.jsonl` dans `data/converted/`.

### 3. Générer les ensembles d'entraînement, validation et test
```bash
python scripts/train_test_val.py votre_fichier.jsonl --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```
Cela générera `train.jsonl`, `validation.jsonl` et `test.jsonl` dans `data/processed/`.

---

##  Entraînement du modèle
### 1. Uploader les données sur Mistral
```bash
python scripts/upload_data.py --train data/processed/train.jsonl --val data/processed/validation.jsonl
```

### 2. Créer un job et lancer l'entraînement
```bash
python scripts/train.py --train_file train.jsonl --val_file validation.jsonl
```
Ce script va lancer l'entraînement et enregistrer les logs sur Weights & Biases.

---

##  Inférence
Testez le modèle fine-tuné sur un prompt (cette inférence se fera sur le dernier modèle dans la liste des modèles fine-tunés):
```bash
python scripts/infer.py --input "votre_prompt"
```

---

##  Évaluation des métriques
L'évaluation réalise l'ensemble des inférences sur le dataset test avec le modèle fine-tuné et le modèle de base. Les scores calculés sont :
- **BLEU** (qualité des réponses par rapport à la réponse attendue)
- **ROUGE-1, ROUGE-2, ROUGE-L** (recouvrement des mots-clés)
- **F1-Score** (précision et rappel combinés)

Lancer l'évaluation :
Ici également l'évaluation se fera sur le dernier modèle fine-tuné
```bash
python scripts/evaluate.py data/processed/test.jsonl mon-modele-finetune
```
Les résultats seront affichés sous forme d'histogrammes.

---

**Auteur :** [aelharra1]


