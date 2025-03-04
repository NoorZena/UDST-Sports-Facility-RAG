import streamlit as st
import os
import numpy as np
import faiss
from mistralai import Mistral
from mistralai.models import UserMessage

# Load Mistral API Key from environment
API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    st.error("⚠️ MISTRAL_API_KEY is missing! Set it in environment variables.")
    st.stop()

# Initialize Mistral AI Client
client = Mistral(api_key=API_KEY)

# Function to generate embeddings using Mistral
def get_text_embedding(text):
    response = client.embeddings.create(model="mistral-embed", inputs=[text])
    return response.data[0].embedding

# Predefined facility descriptions and embeddings
facility_descriptions = [
    "The Sports Hall allows basketball, volleyball, and badminton with 60-minute sessions.",
    "The Running Track is 400m long and allows group running and training sessions.",
    "The Swimming Pool is a 25m heated indoor facility with safety equipment available.",
    "The Multipurpose Hall accommodates futsal, handball, and volleyball with all necessary amenities.",
    "The Tennis Court features a high-quality DecoTurf surface, windbreaks, and privacy screens.",
    "The Padel Courts are designed for extreme weather resistance and ensure privacy for female users.",
    "The Gym includes cardio machines, free weights, and resistance equipment for a full workout session.",
    "The Beach Volleyball Court allows 30 participants per 60-minute session for training and games.",
    "The Indoor Squash Courts encourage players to bring their own equipment for a competitive game."
]

# Generate embeddings for stored descriptions
stored_embeddings = [get_text_embedding(desc) for desc in facility_descriptions]

# Load FAISS Index
def load_faiss_index(embeddings_list):
    d = len(embeddings_list[0])
    index = faiss.IndexFlatL2(d)
    np_embeddings = np.array(embeddings_list).astype('float32')
    index.add(np_embeddings)
    return index

index = load_faiss_index(stored_embeddings)

# Streamlit UI
st.title("🏟️ Sports Facility RAG App")
st.write("This app retrieves sports facility information using **Mistral AI** and **FAISS**.")

# User input
question = st.text_input("Ask a question about the sports facilities:", "What are the ways I can use sports facilities?")

if st.button("Search"):
    st.write(f"🔍 Searching for relevant information about: **{question}**")

    # Get the embedding for the question
    question_embedding = np.array(get_text_embedding(question)).astype('float32').reshape(1, -1)

    # Search FAISS for the most relevant answer
    D, I = index.search(question_embedding, k=1)  # Retrieve top 1 match

    # Display the retrieved answer
    retrieved_text = facility_descriptions[I[0][0]]
    
    st.write("### 📌 Answer:")
    st.success(retrieved_text)
