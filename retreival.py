import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
ENDPOINT = "https://gpt-4o-intern.openai.azure.com/"
MODEL_NAME = "gpt-4.1"
DEPLOYMENT = "gpt-4.1"
SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)

# --- Load Resources ---
def load_knowledge_base():
    """Load the knowledge base if it exists, otherwise return None"""
    try:
        if (os.path.exists("embeddings.npy") and 
            os.path.exists("faiss_index.index") and 
            os.path.exists("chunks.json")):
            
            embeddings = np.load("embeddings.npy")
            index = faiss.read_index("faiss_index.index")
            with open("chunks.json", 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            return embeddings, index, chunks
        else:
            return None, None, None
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return None, None, None

# Try to load existing knowledge base
embeddings, index, chunks = load_knowledge_base()
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# --- Query Function ---
def query_rag_system(query_text, index, embeddings, chunks, embedding_model, show_context=False):
    query_embedding = embedding_model.encode([query_text])
    D, I = index.search(np.float32(query_embedding), k=10)  # Retrieve top 10 chunks

    relevant_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join([f"Document {i+1}:\n"+chunk["chunk"] for i, chunk in enumerate(relevant_chunks)])

    if show_context:
        print("___________________________________________________________")
        print("Context:\n", context)
        print("Question:\n", query_text)
        print("___________________________________________________________")

    augmented_prompt = f"""Please answer the following question based on the context provided. 

    Analyze each document in the context and identify if it contains relevant information to answer the question.
    Focus on the most relevant information and provide a clear, concise answer.
    
    If the context doesn't contain sufficient information to answer the question, please respond with 'I am sorry, but the provided context does not have enough information to answer your question.'
    
    Context:
    {context}
    
    Question: {query_text}
    
    Please provide a direct answer without showing your reasoning process."""

    completion = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "user", "content": augmented_prompt}
        ],
        max_completion_tokens=800,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    response = completion.choices[0].message.content
    return response

if __name__ == '__main__':
    while True:
        user_query = input("Ask a question about the PDF documents (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        response = query_rag_system(user_query, index, embeddings, chunks, embedding_model) # Removed llm_agent parameter
        print(f"\n--- Response from {MODEL_NAME} (via E2E) ---")
        print(response)
        print("\n" + "-"*50 + "\n")


