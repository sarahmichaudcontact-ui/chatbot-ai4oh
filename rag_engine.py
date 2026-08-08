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

FAISS_FOLDER = ""  # index.bin et chunks.pkl sont à la racine du dépôt

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
# 🔹 PROMPT SYSTÈME STABLE (VERSION CORRIGÉE)
# ============================================================

SYSTEM_PROMPT = """
Tu es un assistant pédagogique pour les enseignants du primaire au Maroc.
Tu aides à comprendre et utiliser les informations fournies dans le contexte.
Réponds uniquement en français, de manière simple, claire et adaptée à un enseignant non spécialiste.
Base ta réponse UNIQUEMENT sur le contexte fourni.
Si une information n’est pas présente dans le contexte, dis-le simplement.
Ne change pas de thème sans que l’utilisateur le demande.
Ne mentionne jamais les chunks, les documents ou le fonctionnement interne du système.
Ton ton doit être bienveillant, rassurant et professionnel.
"""

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
