# NLP_Summary_Text

## Description
Cette API permet de générer un résumé de texte en utilisant deux méthodes :
1. **TextRank** : Algorithme basé sur un graphe de similarité entre les phrases.
2. **Transformers (BART)** : Utilisation d'un modèle pré-entraîné fine-tuné pour le résumé de texte.

## Technologies utilisées
- **FastAPI** : Framework web pour créer l'API REST.
- **NLTK** : Librairie de traitement du langage naturel pour le prétraitement du texte.
- **NumPy** : Manipulation des matrices pour le calcul de similarité.
- **NetworkX** : Création du graphe et application de l'algorithme PageRank.
- **Hugging Face Transformers** : Utilisation du modèle pré-entraîné `facebook/bart-large-cnn`.

## Installation
### Prérequis
Assurez-vous d'avoir Python 3.8+ installé.

### Étapes d'installation
```bash
# Cloner le projet
git clone https://github.com/votre-repo/api-text-summary.git
cd api-text-summary

# Créer un environnement virtuel
python -m venv env
source env/bin/activate  # Sur Windows: env\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Lancement de l'API
```bash
uvicorn main:app --reload
```
L'API sera accessible à l'adresse : [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Utilisation
### Endpoints disponibles

#### 1. Résumé de texte
**Endpoint :** `POST /summarize`

**Exemple de requête :**
```json
{
    "text": "Votre texte à résumer ici...",
    "num_sentences": 2,
    "method": "textrank"  // ou "transformers"
}
```

**Exemple de réponse :**
```json
{
    "summary": "Résumé généré...",
    "method": "textrank"
}
```

#### 2. Endpoint de test
**Endpoint :** `GET /`

Répond avec un message de bienvenue.

## Améliorations possibles
- Ajouter un modèle GPT pour un résumé encore plus précis.
- Permettre à l'utilisateur de choisir entre plusieurs modèles pré-entraînés.
- Développer une interface graphique pour rendre l'application plus accessible.
- Utilisation d'un GPU pour les modèles puissants tel que BERT. 

## Auteurs
Projet développé par **PADONOU Mickael** dans le cadre d'un projet NLP en Master Big Data.

## Licence
Ce projet est sous licence MIT.

