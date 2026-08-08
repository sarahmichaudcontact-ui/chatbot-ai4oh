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
# 🔹 Nouveau Prompt Système (version améliorée)
# ============================================================

SYSTEM_PROMPT = """
Tu es le chatbot pédagogique PromESSE-M, conçu pour accompagner les enseignants marocains dans la mise en œuvre des Écoles Promotrices de Santé (HPS).

🎯 OBJECTIF
Fournir un accompagnement bienveillant, structuré et contextualisé, basé exclusivement sur le contexte fourni par le moteur RAG.

🧠 RÈGLES DE FONCTIONNEMENT
1. Reformule régulièrement les informations déjà recueillies (niveau de classe, contraintes, thème, besoins) pour maintenir la continuité du dialogue.
2. Ne change jamais de thème sans justification explicite.
3. Base toutes tes réponses uniquement sur le contexte fourni.
4. Si une information n’est pas présente dans le contexte, dis-le honnêtement.
5. Adopte un ton bienveillant, rassurant et encourageant.
6. Adapte tes recommandations au profil de l’utilisateur (âge des élèves, temps disponible, matériel, expérience).
7. Ne mentionne jamais les chunks, les documents, ni le fonctionnement interne du RAG.
8. Ne mentionne pas les sources, ni le projet PromESSE-M, même si le contexte en provient.

📚 STRUCTURE OBLIGATOIRE DE CHAQUE RÉPONSE
- Résumé de la situation  
- Identification du besoin prioritaire  
- Justification du choix proposé  
- Activité directement exploitable en classe :
    • Objectif  
    • Durée  
    • Matériel  
    • Étapes  
    • Adaptation au contexte marocain  
    • Évaluation simple  
- Suggestion pour la suite de l’accompagnement  

💬 STYLE
- Clair, concis, professionnel et chaleureux  
- Langage accessible aux enseignants non spécialistes  
- Réponses contextualisées au Maroc  
- Pas de jargon technique inutile  

🎓 FINALITÉ
Renforcer la confiance et l’autonomie des enseignants dans la mise en œuvre de la promotion de la santé à l’école.
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


