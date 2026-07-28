import streamlit as st
from rag_engine import chatbot_rag

st.title("Promesse-M Bot: Mon assistant en promotion de la santé")

# -----------------------------
# MÉMOIRE DE CONVERSATION
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# CHAMP DE SAISIE
# -----------------------------
question = st.text_input("Pose ta question :")

if question:
    # Ajouter la question dans l'historique
    st.session_state.messages.append({"role": "user", "content": question})

    # Appeler ton moteur RAG (question seule)
    reponse, sources = chatbot_rag(question)

    # Ajouter la réponse dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": reponse})


# -----------------------------
# AFFICHAGE DE LA CONVERSATION
# -----------------------------
st.write("### Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"👤 Vous : {msg['content']}")
    else:
        st.write(f"🤖 Promesse-M Bot : {msg['content']}")


# -----------------------------
# BOUTON POUR EFFACER LA CONVERSATION
# -----------------------------
if st.button("Effacer la conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

