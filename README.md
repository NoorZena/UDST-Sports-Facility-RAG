#  Sports Facility RAG using Mistral AI

A Retrieval-Augmented Generation (RAG) pipeline for querying sports facility details using web scraping, FAISS for embeddings storage, and Mistral AI for generating responses.

##  Features

-  **Web Scraping**: Extracts UDST Sports Facility data using `BeautifulSoup`
-  **Mistral AI Embeddings**: Converts facility descriptions into vector embeddings
- = **FAISS Vector Search**: Stores embeddings for efficient retrieval
-  **AI-Powered Querying**: Allows users to ask **natural language questions** about facilities
-  **Automated Text Generation**: Retrieves relevant information and generates an answer using Mistral LLM

##  Installation

1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/Sports-Facility-RAG.git
cd Sports-Facility-RAG

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Set Up Your Mistral API Key
import os
os.environ["MISTRAL_API_KEY"] = "YOUR_API_KEY"

4️⃣ Run the Jupyter Notebook
Open RAG with Mistral.ipynb and run all cells step by step.

📝 Usage
Run the Web Scraping section to extract sports facility details.
Process the text and generate embeddings.
Store embeddings in FAISS for efficient retrieval.
Ask a question like: 
question = "What are the ways I can use sports facilities?"
The model will return a relevant response using retrieved knowledge.


