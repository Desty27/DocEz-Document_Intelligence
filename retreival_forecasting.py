import os
import faiss
import numpy as np
import json
import openai
import requests
import time
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

API_TOKEN_ENV = "E2E_API_TOKEN"


def _get_api_token(optional: bool = False):
    token = os.getenv(API_TOKEN_ENV)
    if token:
        return token
    if optional:
        return None
    raise RuntimeError(
        "E2E_API_TOKEN environment variable is not set. Provide it in your environment or .env file."
    )

# --- Model configuration ---
MODELS = {
    "DeepSeek R1": {
        "deployment": "deepseek_r1",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/deepseek_r1/v1/",
        "type": "chat"
    },
    "GPT OSS 120B": {
        "deployment": "gpt_oss_120b",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/gpt_oss_120b/v1/",
        "type": "chat"
    },
    "Phi 4": {
        "deployment": "phi_4",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/phi_4/v1/",
        "type": "chat"
    },
    "Gemma 7B": {
        "deployment": "gemma_7b",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/gemma_7b/v1/",
        "type": "chat"
    },
    # --- THIS IS THE UPDATED MODEL ---
    "Llama3.1 8b (Explain)": {
        "deployment": "llama3_1_8b_instruct",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/llama3_1_8b_instruct/v1/",
        "type": "chat"
    }
}

API_TOKEN = os.getenv(API_TOKEN_ENV)  # Store your token in .env


# --- Load RAG Knowledge Base ---
def load_knowledge_base():
    """Load FAISS embeddings and chunks if available"""
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


embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# --- Safe request with retry ---
def safe_completion_request(client, model_info, prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model_info["deployment"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7,
                top_p=0.9
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return "⚠️ Server unavailable. Please try again later."

# --- UPDATED FUNCTION FOR FORECASTING EXPLANATION ---
def get_forecasting_explanation(
    date_col,
    target_col,
    model_errors,
    all_columns=None,
):
    """
    Calls the Llama3.1 LLM to generate a simple explanation of the forecasting process.
    """
    # --- THIS LINE IS CHANGED ---
    model_name = "Llama3.1 8b (Explain)" 
    model_info = MODELS.get(model_name)
    
    if not model_info:
        return "Explanation model 'Llama3.1 8b (Explain)' not configured in retreival.py."
    
    # Convert model_errors dict to a simple string
    error_summary = "\n".join([f"- {model}: {error}" for model, error in model_errors.items()])
    
    # Build the prompt for the LLM
    if all_columns is None:
        columns_text = "Columns list not provided."
    else:
        if isinstance(all_columns, (list, tuple)):
            columns_render = list(all_columns)
        else:
            try:
                columns_render = list(all_columns)
            except TypeError:
                columns_render = [str(all_columns)]
        columns_text = f"All columns in the uploaded file: {columns_render}"

    prompt = f"""
    You are a helpful data analyst. Your job is to explain a complex forecasting process to a non-technical user in simple points.

    Here is the information about the forecast that was just run:

    1. DATASET:
       - {columns_text}
       
    2. PREPROCESSING:
       - Selected Date Column (X-Axis): '{date_col}'
       - Selected Target Column to Forecast (Y-Axis): '{target_col}'
       - Process: The app converted the '{date_col}' column to a standard datetime, set it as the index, and resampled all data to a daily average to make it clean.
       
    3. MODEL ACCURACY (MAPE - Lower is better):
    {error_summary}

    Please write a short, simple, step-by-step summary based *only* on the information above.

    Your summary must include:
    1. A brief "About Your Data" section (mention the target column).
    2. A brief "How We Prepared It" section (mention the date column).
    3. A final "Model Recommendation" section. In this section, you MUST recommend the model with the LOWEST error percentage (MAPE) from the list and explain that it's recommended because it was the most accurate on the historical data.

    Keep it simple and use bullet points. Start with "Here's a simple breakdown of what just happened:".
    """
    
    try:
        token = API_TOKEN or _get_api_token(optional=True)
        if not token:
            return "E2E_API_TOKEN is not configured. Set it in your environment to enable explanations."
        client = openai.OpenAI(api_key=token, base_url=model_info["base_url"])  # type: ignore[attr-defined]
        
        # We use a non-streaming call here to get the full report at once.
        completion = client.chat.completions.create(
            model=model_info["deployment"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.5,
            top_p=1,
            stream=False # Use False for a single report
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling explanation LLM: {e}")
        return f"Error generating explanation: {e}"

# --- RAG + LLM Query ---
def query_rag_system(query_text, index, embeddings, chunks, embedding_model, model_name="DeepSeek R1", show_context=False):
    # Step 1: Retrieve relevant chunks
    query_embedding = embedding_model.encode([query_text])
    D, I = index.search(np.float32(query_embedding), k=10)
    relevant_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join([f"Document {i+1}:\n"+chunk["chunk"] for i, chunk in enumerate(relevant_chunks)])

    if show_context:
        print("------ CONTEXT ------")
        print(context)
        print("--------------------")

    # Step 2: Augment prompt for chat models
    augmented_prompt = f"""Please answer the following question based on the context provided. 

Context:
{context}

Question: {query_text}

Provide a clear and concise answer. If the context is insufficient, respond with 'I am sorry, but the provided context does not have enough information to answer your question.'"""

    model_info = MODELS.get(model_name)
    if model_info is None:
        return "Unsupported model type."

    response = "Unsupported model type."

    if model_info["type"] == "chat":
        token = API_TOKEN or _get_api_token(optional=True)
        if not token:
            return "E2E_API_TOKEN is not configured. Set it in your environment to enable forecasting responses."
        client = openai.OpenAI(api_key=token, base_url=model_info["base_url"])  # type: ignore[attr-defined]
        response = safe_completion_request(client, model_info, augmented_prompt)
    elif model_info["type"] == "translation":
        # (Translation logic unchanged)
        payload = {
            "inputs": [
                {"name": "prompt", "shape": [1, 1], "datatype": "BYTES", "data": [query_text]},
                {"name": "INPUT_LANGUAGE_ID", "shape": [1, 1], "datatype": "BYTES", "data": ["en"]},
                {"name": "OUTPUT_LANGUAGE_ID", "shape": [1, 1], "datatype": "BYTES", "data": ["hi"]}
            ]
        }
        token = API_TOKEN or _get_api_token(optional=True)
        if not token:
            return "E2E_API_TOKEN is not configured. Set it in your environment to enable forecasting responses."
        headers = {"authorization": f"Bearer {token}", "content-type": "application/json"}
        try:
            resp = requests.post(model_info["base_url"], headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                response_json = resp.json()
                response = response_json['outputs'][0]['data'][0]
            else:
                response = f"Error: {resp.status_code} - {resp.text}"
        except Exception as e:
            response = f"Error during translation request: {e}"

    return response


if __name__ == "__main__":
    embeddings, index, chunks = load_knowledge_base()
    if not embeddings:
        print("Knowledge base not found. Process documents first.")
        exit()

    while True:
        user_query = input("Ask a question (type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        selected_model = input(f"Select model ({', '.join(MODELS.keys())}): ")
        if selected_model not in MODELS:
            print("Invalid model. Using default DeepSeek R1.")
            selected_model = "DeepSeek R1"

        response = query_rag_system(user_query, index, embeddings, chunks, embedding_model, model_name=selected_model)
        print(f"\n--- Response from {selected_model} ---")
        print(response)
        print("\n" + "-"*50 + "\n")

