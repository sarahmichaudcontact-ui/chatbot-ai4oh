import streamlit as st
from rag_engine import chatbot_rag

st.title("Promesse-M Bot: Mon assistant en promotion de la santé")

question = st.text_input("Pose ta question :")

if question:
    reponse, sources = chatbot_rag(question)
    st.write("Vous :", question)
    st.write("🤖 Promesse-M Bot :", reponse)

