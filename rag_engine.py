# ============================================================
# 🔹 Moteur RAG pour Promesse-M Bot (version Streamlit Cloud)
# ============================================================

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# ============================================================
# 🔹 Chargement du corpus et de l'index FAISS
# ============================================================

# ⚠️ IMPORTANT :
# Dans ton dépôt GitHub, index.bin et chunks.pkl sont à la racine :
# - chatbot-ai4oh/index.bin
# - chatbot-ai4oh/chunks.pkl
#
# Donc FAISS_FOLDER doit être vide :

FAISS_FOLDER = ""

# Charger l’index FAISS
index = faiss.read_index("index.bin")

# Charger les chunks
with open("chunks.pkl", "rb") as f:
    tous_les_chunks = pickle.load(f)

# ============================================================
# 🔹 Modèle d'embedding
# ============================================================

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ============================================================
# 🔹 Client Groq (clé API sécurisée via Streamlit)
# ============================================================

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ============================================================
# 🔹 Prompt système
# ============================================================

SYSTEM_PROMPT = """Tu es un assistant pédagogique pour les enseignants 
du primaire au Maroc.
Tu aides les enseignants à mettre en œuvre les principes des Écoles 
Promotrices de Santé.
Réponds uniquement en français, de façon claire et pratique.
Base tes réponses UNIQUEMENT sur le contexte fourni.
Ne mentionne jamais les sources, documents ou chunks dans ta réponse.
Ne mentionne jamais le projet PromESSE-M.
Réponds directement à l'enseignant comme un conseiller pédagogique.
Si tu ne trouves pas l'information dans le contexte, dis honnêtement 
que tu ne disposes pas de cette information."""

# ============================================================
# 🔹 Fonction principale RAG
# ============================================================

def chatbot_rag(question, k=5):
    """
    Fonction principale du moteur RAG.
    Recherche les chunks pertinents et génère une réponse contextualisée.
    """

    # 1️⃣ Encoder la question
    question_vec = embed_model.encode([question])

    # 2️⃣ Recherche FAISS
    distances, indices = index.search(np.array(question_vec), k)
    chunks_pertinents = [tous_les_chunks[i]["texte"] for i in indices[0]]

    # 3️⃣ Construire le prompt
    contexte = "\n\n".join(chunks_pertinents)
    prompt = (
        f"{SYSTEM_PROMPT}\n\nContexte:\n{contexte}\n\n"
        f"Question: {question}\nRéponse:"
    )

    # 4️⃣ Appel au modèle Groq
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    # 5️⃣ Retourner la réponse + sources
    return response.choices[0].message.content, chunks_pertinents





