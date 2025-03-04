# Fine_tuning_mistral7B

# Fine-tuning Mistral 7B avec un dataset médical

Ce projet permet de fine-tuner le modèle **Mistral 7B** sur un dataset médical. Il comprend des scripts pour la préparation des données, l'entraînement, l'inférence et l'évaluation du modèle.

---

## 📁 Structure du projet
```
/MonProjet-FineTuning
│── /data
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│── /scripts
│   ├── preprocess.py  # Conversion et préparation des données
│   ├── upload_data.py  # Upload des datasets sur Mistral
│   ├── train.py  # Fine-tuning du modèle
│   ├── infer.py  # Inférence sur de nouvelles données
│   ├── evaluate.py  # Calcul des métriques d’évaluation
│── /notebooks
│   ├── version3.ipynb  # Notebook Google Colab complet
│── requirements.txt  # Liste des dépendances
│── README.md  # Documentation du projet
│── .gitignore  # Exclusion des fichiers inutiles
```

---

## 🚀 Installation
### 1. Cloner le repo
```bash
git clone https://github.com/ton-utilisateur/MonProjet-FineTuning.git
cd MonProjet-FineTuning
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Préparer les données
Assurez-vous que votre dataset est au format CSV, puis exécutez :
```bash
python scripts/preprocess.py
```
Cela va générer trois fichiers : `train.jsonl`, `validation.jsonl` et `test.jsonl`.

### 4. Uploader les données sur Mistral
```bash
python scripts/upload_data.py
```

### 5. Lancer l'entraînement
```bash
python scripts/train.py
```

### 6. Faire une inférence
```bash
python scripts/infer.py --input "Texte à tester"
```

### 7. Évaluer le modèle
```bash
python scripts/evaluate.py
```

---

## 📊 Métriques d'évaluation
Le script `evaluate.py` calcule les scores suivants :
- **BLEU**
- **ROUGE-1, ROUGE-2, ROUGE-L**
- **F1-Score**

---

## 📌 Notes
- Ce projet utilise **Weight & Biases (W&B)** pour le suivi des performances.
- Vous pouvez modifier les hyperparamètres dans `train.py`.
- Pour exécuter le projet sous Google Colab, utilisez le notebook `version3.ipynb`.

---

## ✨ Contributions
Les contributions sont les bienvenues ! N'hésitez pas à proposer des améliorations via une pull request.

## 📜 Licence
Ce projet est sous licence MIT. Vous êtes libre de l'utiliser et de le modifier.

---

**Auteur :** [Ton Nom]

