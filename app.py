import streamlit as st

st.title("Promesse-M Bot : Mon assistant en promotion de la santé")

user_input = st.text_input("Pose ta question :")

if user_input:
    st.write("👤 Vous :", user_input)
    st.write("🤖 Prototype :", "Merci ! Le chatbot sera bientôt connecté au moteur RAG.")
