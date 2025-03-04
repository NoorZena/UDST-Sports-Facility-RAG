import streamlit as st

st.title("Sports Facility RAG App")
st.write("This app allows you to retrieve sports facility information using Mistral AI and FAISS.")

question = st.text_input("Ask a question about the sports facilities:")
if question:
    st.write(f"Searching for relevant information about: {question}")
    # Add FAISS search and response generation logic here.
