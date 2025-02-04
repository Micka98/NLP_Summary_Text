import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI()

# Modèle pour la requête utilisateur
class TextInput(BaseModel):
    text: str
    num_sentences: int = 2
    method: str = "textrank"  # "textrank" ou "transformers"

# Chargement d'un modèle pré-entraîné pour le résumé
transformers_model = "facebook/bart-large-cnn"
summary_pipeline = pipeline("summarization", model=transformers_model)

# Fonction de nettoyage du texte
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

# Fonction pour nettoyer les guillemets et apostrophes
def clean_text_input(text: str) -> str:
    text = text.replace("’", "'")  # Convertir les apostrophes typographiques
    text = text.replace('"', '\"')  # Échapper les guillemets doubles
    return text

# Fonction pour générer un résumé à l'aide de TextRank
def generate_summary_textrank(text, num_sentences=2):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        raise ValueError("Le texte fourni ne contient pas de phrases valides.")

    cleaned_sentences = [clean_text(sentence) for sentence in sentences]

    # Calcule de la similarité entre les phrases
    similarity_matrix = np.zeros((len(cleaned_sentences), len(cleaned_sentences)))
    for i, sent1 in enumerate(cleaned_sentences):
        for j, sent2 in enumerate(cleaned_sentences):
            if i != j:
                words1 = set(word_tokenize(sent1))
                words2 = set(word_tokenize(sent2))
                similarity_matrix[i][j] = len(words1 & words2) / (len(words1 | words2) + 1e-5)

    # Création du graphe et appliquation de PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Trie des phrases par score
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)

    # Sélection des phrases les plus importantes
    summary = ' '.join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    return summary

# Fonction pour générer un résumé à l'aide de Transformers
def generate_summary_transformers(text):
    try:
        result = summary_pipeline(text, max_length=130, min_length=30, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        raise ValueError("Erreur lors de la génération du résumé avec Transformers : " + str(e))

@app.post("/summarize")
async def summarize_text(input_data: TextInput):
    try:
        input_data.text = clean_text_input(input_data.text)  # Nettoyer le texte avant traitement
        
        if input_data.method == "textrank":
            summary = generate_summary_textrank(input_data.text, input_data.num_sentences)
        elif input_data.method == "transformers":
            summary = generate_summary_transformers(input_data.text)
        else:
            raise HTTPException(status_code=400, detail="Méthode non valide. Utilisez 'textrank' ou 'transformers'.")

        return {"summary": summary, "method": input_data.method}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Une erreur est survenue lors de la génération du résumé.")

@app.get("/")
async def root():
    return {"message": "Bienvenue dans l'API de résumé de texte avec TextRank et Transformers. Utilisez l'endpoint /summarize pour générer un résumé."}
