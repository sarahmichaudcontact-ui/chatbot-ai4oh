import streamlit as st
from rag_engine import ask  # ton moteur RAG

st.set_page_config(page_title="Chatbot AI4OH", page_icon="🤖")

# -----------------------------
# 1. MÉMOIRE DE CONVERSATION
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# 2. TITRE
# -----------------------------
st.title("🤖 Chatbot AI4OH")
st.write("Posez vos questions sur votre mémoire, votre chatbot ou l’IA One Health.")


# -----------------------------
# 3. CHAMP DE SAISIE
# -----------------------------
user_input = st.text_input("Votre question :")

if user_input:
    # Ajouter la question dans l’historique
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Appeler ton moteur RAG AVEC l’historique complet
    response = ask(st.session_state.messages)

    # Ajouter la réponse dans l’historique
    st.session_state.messages.append({"role": "assistant", "content": response})


# -----------------------------
# 4. AFFICHAGE DE LA CONVERSATION
# -----------------------------
st.write("### Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"👤 **Vous :** {msg['content']}")
    else:
        st.write(f"🤖 **Chatbot :** {msg['content']}")


# -----------------------------
# 5. BOUTON POUR EFFACER LA CONVERSATION
# -----------------------------
if st.button("Effacer la conversation"):
    st.session_state.messages = []
    st.experimental_rerun()
