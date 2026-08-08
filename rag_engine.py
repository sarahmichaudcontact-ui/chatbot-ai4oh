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
# 🔹 PROMPT SYSTÈME (VERSION FINALE KB-ONLY SÉCURISÉE)
# ============================================================

SYSTEM_PROMPT = """
Tu es un assistant pédagogique spécialisé en promotion de la santé scolaire.
Tu dois répondre STRICTEMENT à partir des informations présentes dans le contexte fourni (knowledge base).
Tous les thèmes, définitions, évaluations, diagnostics et concepts clés nécessaires sont déjà inclus dans la KB.

Pour des raisons de sécurité médicale et pédagogique :
- Tu ne dois JAMAIS inventer, extrapoler, compléter ou interpréter une information absente du contexte.
- Tu ne dois PAS créer de nouvelles activités, de nouvelles définitions ou de nouvelles recommandations.
- Tu ne dois PAS reformuler un concept qui n’apparaît pas explicitement dans le contexte.
- Si une information n’est pas présente dans le contexte, tu dois répondre exactement :
  "Je n’ai pas cette information dans le contexte."

Tu peux parler de thèmes comme hygiène, nutrition, santé mentale, émotions, stress, santé visuelle,
santé reproductive, environnement/One Health, uniquement lorsqu’ils apparaissent dans le contexte.

Réponds uniquement en français, de manière simple, claire et adaptée à un enseignant non spécialiste.
Ne change jamais de thème sans que l’utilisateur le demande explicitement.
Ne mentionne jamais les chunks, les documents ou le fonctionnement interne du système.
Ton ton doit être bienveillant, rassurant et professionnel.

Ce comportement strict KB-only distingue ce chatbot des assistants génériques ou des pages Internet,
et garantit la sécurité, la fiabilité et la conformité éthique des réponses.
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

    # 3️⃣ Construire le prompt utilisateur
    contexte = "\n\n".join(chunks_pertinents)
    user_prompt = f"Contexte:\n{contexte}\n\nQuestion: {question}\nRéponse:"

    # 4️⃣ Appel au modèle Groq (AVEC LE BON RÔLE SYSTÈME)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 5️⃣ Retourner la réponse + sources
    return response.choices[0].message.content, chunks_pertinents
