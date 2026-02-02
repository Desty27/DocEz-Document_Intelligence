import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import openai
from dotenv import load_dotenv
import requests
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, cast
from helpers import generate_followup_questions, build_improved_prompt

load_dotenv()

E2E_API_TOKEN_ENV = "E2E_API_TOKEN"
SARVAM_API_KEY_ENV = "SARVAM_API_KEY"


def _get_e2e_token(optional: bool = False) -> Optional[str]:
    token = os.getenv(E2E_API_TOKEN_ENV)
    if token:
        return token
    if optional:
        return None
    raise RuntimeError(
        "E2E_API_TOKEN environment variable is not set. Provide it in your environment or .env file."
    )


def _get_sarvam_api_key(optional: bool = False) -> Optional[str]:
    api_key = os.getenv(SARVAM_API_KEY_ENV)
    if api_key:
        return api_key
    if optional:
        return None
    raise RuntimeError(
        "SARVAM_API_KEY environment variable is not set. Provide it in your environment or .env file."
    )


# Model configurations
MODELS = {
    "DeepSeek R1": {
        "model_name": "deepseek_r1",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/deepseek_r1/v1/",
        "deployment": "deepseek_r1"
    },
    "GPT OSS 120B": {
        "model_name": "gpt_oss_120b", 
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/gpt_oss_120b/v1/",
        "deployment": "gpt_oss_120b"
    },
    "Llama 4 Scout": {
        "model_name": "llama_4_scout_17b_16e_instruct",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/llama_4_scout_17b_16e_instruct/v1/",
        "deployment": "llama_4_scout_17b_16e_instruct"
    },
    "Gemma 7B": {
        "model_name": "gemma_7b",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/gemma_7b/v1/",
        "deployment": "gemma_7b"
    },
    "Phi 4": {
        "model_name": "phi_4",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/phi_4/v1/",
        "deployment": "phi_4"
    },
    "Llama 3.1 8B": {
        "model_name": "llama3_1_8b_instruct",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/llama3_1_8b_instruct/v1/",
        "deployment": "llama3_1_8b_instruct"
    },
    "Qwen2 VL 72B": {
        "model_name": "qwen2_vl_72b_instruct",
        "base_url": "https://infer.e2enetworks.net/project/p-5915/genai/qwen2_vl_72b_instruct/v1/",
        "deployment": "qwen2_vl_72b_instruct"
    },
    "SarvamAI": {
        "model_name": "sarvam_ai",
        "base_url": "sarvam",  # Special identifier for SarvamAI
        "deployment": "sarvam_ai",
        "api_key": _get_sarvam_api_key(optional=True)
    }
}

def check_model_status(model_name: str) -> bool:
    """Check if a model is online by making a simple API call"""
    try:
        config = MODELS[model_name]
        
        # Special handling for SarvamAI
        if model_name == "SarvamAI":
            # SarvamAI uses a different API endpoint
            import requests
            api_key = config.get("api_key") or _get_sarvam_api_key(optional=True)
            if not api_key:
                print("SarvamAI API key missing; mark model offline.")
                return False
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "sarvam-1",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }
            response = requests.post(
                "https://api.sarvam.ai/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            return response.status_code == 200
        else:
            # Standard OpenAI-compatible API
            token = _get_e2e_token(optional=True)
            if not token:
                print(f"E2E_API_TOKEN missing; {model_name} marked offline.")
                return False
            client = openai.OpenAI(  # type: ignore[attr-defined]
                api_key=token,
                base_url=config["base_url"]
            )
            
            # Try a simple completion to check if model is online
            response = client.chat.completions.create(
                model=config["deployment"],
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                timeout=10
            )
            return True
    except Exception as e:
        print(f"Model {model_name} appears offline: {e}")
        return False

def get_available_models() -> Dict[str, bool]:
    """Get the online status of all models"""
    model_status = {}
    for model_name in MODELS.keys():
        model_status[model_name] = check_model_status(model_name)
    return model_status

class SarvamAIClient:
    """Custom client for SarvamAI API that mimics OpenAI structure"""
    
    class Chat:
        class Completions:
            def __init__(self, client):
                self.client = client
            
            def create(self, model, messages, max_tokens=800, temperature=0.7, **kwargs):
                import requests
                
                headers = {
                    "Authorization": f"Bearer {self.client.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "sarvam-1",  # SarvamAI's model identifier
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                response = requests.post(
                    f"{self.client.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise Exception(f"SarvamAI API error: {response.status_code} - {response.text}")
                
                result = response.json()
                
                # Create a mock response object similar to OpenAI's structure
                class MockChoice:
                    def __init__(self, content):
                        self.message = MockMessage(content)
                
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                
                class MockResponse:
                    def __init__(self, content):
                        self.choices = [MockChoice(content)]
                
                # Extract content from SarvamAI response
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return MockResponse(content)
        
        def __init__(self, client):
            self.completions = self.Completions(client)
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sarvam.ai"
        self.chat = self.Chat(self)

def create_client_for_model(model_name: str):
    """Create a client for the specified model"""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODELS[model_name]
    
    # Special handling for SarvamAI
    if model_name == "SarvamAI":
        api_key = config.get("api_key") or _get_sarvam_api_key()
        return SarvamAIClient(cast(str, api_key))
    else:
        # Standard OpenAI-compatible API
        token = _get_e2e_token()
        return openai.OpenAI(  # type: ignore[attr-defined]
            api_key=token,
            base_url=config["base_url"]
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

# --- InLegalBERT Integration ---
class InLegalBERTEmbedder:
    """Custom embedder using InLegalBERT for legal document understanding"""
    
    def __init__(self, model_name="law-ai/InLegalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
    def encode(self, texts):
        """Encode texts using InLegalBERT and return embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        with torch.no_grad():
            for text in texts:
                # Tokenize text
                encoded_input = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                # Get model output
                output = self.model(**encoded_input)
                
                # Use mean pooling of last hidden state
                last_hidden_state = output.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                
                # Mean pooling
                masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
                sum_embeddings = masked_embeddings.sum(dim=1)
                seq_length = attention_mask.sum(dim=1, keepdim=True)
                embedding = sum_embeddings / seq_length
                
                embeddings.append(embedding.squeeze().numpy())
        
        return np.array(embeddings)

# Embedding model configuration
EMBEDDING_MODELS = {
    "sentence-transformers": SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
    "inlegalbert": InLegalBERTEmbedder("law-ai/InLegalBERT")
}

# Default embedding model - can be changed based on document type
DEFAULT_EMBEDDING_MODEL = "sentence-transformers"  # Change to "inlegalbert" for legal documents

# Try to load existing knowledge base
embeddings, index, chunks = load_knowledge_base()
embedding_model = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]


# --- Enhanced Query Functions ---
def detect_legal_content(text: str) -> bool:
    """Detect if text contains legal content based on keywords"""
    legal_keywords = [
        'contract', 'agreement', 'legal', 'law', 'court', 'judge', 'plaintiff', 'defendant',
        'liability', 'damages', 'breach', 'clause', 'statute', 'regulation', 'compliance',
        'jurisdiction', 'litigation', 'precedent', 'case law', 'constitutional', 'amendment',
        'ordinance', 'bylaw', 'tort', 'criminal', 'civil', 'administrative', 'judicial'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in legal_keywords)

def get_optimal_embedding_model(query_text: Optional[str] = None, chunks: Optional[List] = None):
    """Determine the best embedding model based on content analysis"""
    if query_text and detect_legal_content(query_text):
        return EMBEDDING_MODELS["inlegalbert"]
    
    if chunks:
        legal_chunk_count = sum(1 for chunk in chunks[:10] if detect_legal_content(chunk.get('chunk', '')))
        if legal_chunk_count > 5:  # Majority are legal documents
            return EMBEDDING_MODELS["inlegalbert"]
    
    return EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]

def query_rag_system(
    query_text,
    index,
    embeddings,
    chunks,
    embedding_model,
    selected_model="DeepSeek R1",
    show_context=False,
    use_legal_model=None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    history_window: int = 5,
):
    # Determine optimal embedding model if not specified
    if use_legal_model is None:
        optimal_model = get_optimal_embedding_model(query_text, chunks)
        if optimal_model != embedding_model:
            print(f"🏛️ Detected legal content - switching to InLegalBERT for better accuracy")
            embedding_model = optimal_model
    elif use_legal_model:
        embedding_model = EMBEDDING_MODELS["inlegalbert"]
        print(f"🏛️ Using InLegalBERT for legal document analysis")
    
    query_embedding = embedding_model.encode([query_text])
    D, I = index.search(np.float32(query_embedding), k=10)  # Retrieve top 10 chunks

    relevant_chunks = [chunks[i] for i in I[0]]

    # Build context and metadata list (try to include filename and any page info if present)
    context_pieces = []
    metadata = []
    for idx, chunk in enumerate(relevant_chunks):
        chunk_text = chunk.get('chunk', '')
        filename = chunk.get('filename', 'unknown')
        # If chunk contains a page marker, try to find it (simple heuristic)
        page_hint = ''
        m = None
        if 'page' in chunk:
            page_hint = f"page: {chunk.get('page')}"
        else:
            # look for 'page' like patterns inside chunk text
            m = None
        context_pieces.append(f"[{filename}] {chunk_text}")
        metadata.append(f"filename: {filename}")

    context = "\n\n".join(context_pieces)

    if show_context:
        print("___________________________________________________________")
        print("Context:\n", context)
        print("Question:\n", query_text)
        print("Metadata:\n", metadata)
        print("___________________________________________________________")

    # Enhanced prompt for legal content
    is_legal_query = detect_legal_content(query_text)
    if is_legal_query:
        augmented_prompt = f"""Please answer the following legal question based on the context provided. 

        As a legal analysis system, carefully examine each document for relevant legal principles, statutes, regulations, case law, or contractual provisions.
        Provide precise legal analysis and cite specific sections when available.
        
        If the context doesn't contain sufficient legal information to answer the question, please respond with 'I am sorry, but the provided context does not have enough legal information to answer your question. Please consult with a qualified legal professional.'
        
        Context:
        {context}
        
        Legal Question: {query_text}
        
        Please provide a comprehensive legal analysis with relevant citations where available."""
    else:
        augmented_prompt = f"""Please answer the following question based on the context provided. 

        Analyze each document in the context and identify if it contains relevant information to answer the question.
        Focus on the most relevant information and provide a clear, concise answer.
        
        If the context doesn't contain sufficient information to answer the question, please respond with 'I am sorry, but the provided context does not have enough information to answer your question.'
        
        Context:
        {context}
        
        Question: {query_text}
        
        Please provide a direct answer without showing your reasoning process."""

    # If conversation history is provided, include the last N exchanges as part of the question payload
    # conversation_history is expected to be a list of dicts: [{"role":"user","content":"..."}, ...]
    convo_block = ''
    if conversation_history:
        # Take the last history_window exchanges (each exchange is user+assistant)
        # We'll keep it simple and include the last 2*history_window messages
        max_msgs = history_window * 2
        recent = conversation_history[-max_msgs:]
        parts = []
        for m in recent:
            role = m.get('role', 'user')
            content = m.get('content', '')
            parts.append(f"{role.capitalize()}: {content}")
        convo_block = "\n".join(parts)

    # Build the final improved prompt which instructs the model to cite metadata
    # Prepend conversation history to the current question to provide context for follow-ups
    if convo_block:
        combined_question = f"Conversation history:\n{convo_block}\n\nCurrent question: {query_text}"
    else:
        combined_question = query_text

    final_prompt = build_improved_prompt(combined_question, context, metadata)

    # Create client for the selected model
    client = create_client_for_model(selected_model)
    model_config = MODELS[selected_model]

    # All clients now use the same interface thanks to our custom SarvamAI client
    completion = client.chat.completions.create(
        model=model_config["deployment"],
        messages=[
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=800,
        temperature=0.7,
        top_p=0.9 if selected_model != "SarvamAI" else None,
        frequency_penalty=0.0 if selected_model != "SarvamAI" else None,
        presence_penalty=0.0 if selected_model != "SarvamAI" else None
    )
    response = completion.choices[0].message.content

    # Generate follow-up questions locally using heuristics to avoid extra LLM calls
    try:
        followups = generate_followup_questions(response, query_text, max_qs=3)
    except Exception:
        followups = []

    # Return both response and follow-ups separately; the model answer will NOT include follow-ups
    return {"answer": response, "followups": followups}

def delete_file_from_knowledge_base(filename: str) -> bool:
    """Delete a specific file and its chunks from the knowledge base"""
    try:
        # Load existing data
        if not all(os.path.exists(f) for f in ["chunks.json", "embeddings.npy", "faiss_index.index"]):
            return False
        
        with open("chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        embeddings = np.load("embeddings.npy")
        
        # Find chunks to remove
        indices_to_keep = []
        chunks_to_keep = []
        
        for i, chunk in enumerate(chunks):
            if chunk.get('filename', '') != filename:
                indices_to_keep.append(i)
                chunks_to_keep.append(chunk)
        
        if len(chunks_to_keep) == len(chunks):
            # File not found
            return False
        
        # Update embeddings
        if len(chunks_to_keep) > 0:
            embeddings_to_keep = embeddings[indices_to_keep]
            
            # Rebuild FAISS index
            dimension = embeddings_to_keep.shape[1]
            new_index = faiss.IndexFlatL2(dimension)
            # Ensure embeddings are in the correct format for faiss
            embeddings_float32 = embeddings_to_keep.astype('float32')
            if embeddings_float32.ndim == 1:
                embeddings_float32 = embeddings_float32.reshape(1, -1)
            new_index.add(embeddings_float32)
            
            # Save updated data
            np.save("embeddings.npy", embeddings_to_keep)
            faiss.write_index(new_index, "faiss_index.index")
            
            with open("chunks.json", 'w', encoding='utf-8') as f:
                json.dump(chunks_to_keep, f, ensure_ascii=False, indent=2)
        else:
            # No chunks left, remove all files
            for file_path in ["embeddings.npy", "faiss_index.index", "chunks.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Also try to remove the physical file
        data_file_path = os.path.join("data", filename)
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
        
        return True
        
    except Exception as e:
        print(f"Error deleting file from knowledge base: {e}")
        return False

def get_processed_files() -> List[Dict]:
    """Get list of processed files with their chunk counts"""
    try:
        if not os.path.exists("chunks.json"):
            return []
        
        with open("chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        file_counts = {}
        for chunk in chunks:
            filename = chunk.get('filename', 'Unknown')
            if filename in file_counts:
                file_counts[filename] += 1
            else:
                file_counts[filename] = 1
        
        return [{"filename": filename, "chunks": count} for filename, count in file_counts.items()]
        
    except Exception:
        return []

if __name__ == '__main__':
    # Check available models
    print("Checking model availability...")
    available_models = get_available_models()
    online_models = [model for model, status in available_models.items() if status]
    
    if not online_models:
        print("No models are currently online. Please try again later.")
        exit()
    
    print("Available models:")
    for i, model in enumerate(online_models, 1):
        print(f"{i}. {model}")
    
    # Model selection
    selected_model = online_models[0]  # Default to first available
    if len(online_models) > 1:
        try:
            choice = int(input(f"Select model (1-{len(online_models)}, default=1): ")) - 1
            if 0 <= choice < len(online_models):
                selected_model = online_models[choice]
        except (ValueError, IndexError):
            pass
    
    print(f"Using model: {selected_model}")
    
    while True:
        user_query = input("Ask a question about the PDF documents (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        response = query_rag_system(user_query, index, embeddings, chunks, embedding_model, selected_model)
        print(f"\n--- Response from {selected_model} ---")
        print(response)
        print("\n" + "-"*50 + "\n")


