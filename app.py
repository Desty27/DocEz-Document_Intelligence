import streamlit as st
import os
import tempfile
import shutil
from indexing import process_documents
from retreival import query_rag_system
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="GESIL RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #FF9736, #DB791D);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .logo-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .company-logo {
        max-height: 80px;
        max-width: 200px;
        margin-bottom: 1rem;
        object-fit: contain;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
        transition: transform 0.3s ease;
    }
    
    .company-logo:hover {
        transform: scale(1.05);
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #3498db;
        margin: 1rem 0;
    }
    
    .query-section {
        background: #f1f3f5;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .response-section {
        background: #e8f5e8;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #27ae60;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2c3e50);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.8;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def load_logo():
    """Load and encode company logo for display"""
    logo_path = "assets/logo.png"  # You can change this path
    
    # Check if logo file exists
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as logo_file:
                logo_data = base64.b64encode(logo_file.read()).decode()
            return f"data:image/png;base64,{logo_data}"
        except Exception as e:
            print(f"Error loading logo: {e}")
            return None
    else:
        # If no logo file found, you can use a placeholder or skip
        return None

def upload_logo_interface():
    """Interface for uploading a new company logo"""
    with st.expander("🖼️ Update Company Logo", expanded=False):
        st.write("Upload a new company logo (PNG/JPG, max 2MB)")
        
        uploaded_logo = st.file_uploader(
            "Choose logo file",
            type=['png', 'jpg', 'jpeg'],
            help="Recommended: 200x80 pixels, transparent background PNG"
        )
        
        if uploaded_logo is not None:
            if st.button("Update Logo"):
                try:
                    # Ensure assets directory exists
                    os.makedirs("assets", exist_ok=True)
                    
                    # Save the uploaded logo
                    with open("assets/logo.png", "wb") as f:
                        f.write(uploaded_logo.getbuffer())
                    
                    st.success("Logo updated successfully! Refresh the page to see changes.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating logo: {e}")

def initialize_session_state():
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'knowledge_base_ready' not in st.session_state:
        st.session_state.knowledge_base_ready = check_knowledge_base()
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def check_knowledge_base():
    """Check if knowledge base files exist"""
    required_files = ["embeddings.npy", "faiss_index.index", "chunks.json"]
    return all(os.path.exists(f) for f in required_files)

def load_rag_components():
    """Load RAG system components"""
    try:
        embeddings = np.load("embeddings.npy")
        index = faiss.read_index("faiss_index.index")
        with open("chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        return embeddings, index, chunks, embedding_model
    except Exception as e:
        st.error(f"Error loading RAG components: {e}")
        return None, None, None, None

def save_uploaded_files(uploaded_files):
    """Save uploaded files to data directory"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    saved_files = []
    for uploaded_file in uploaded_files:
        try:
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {e}")
    
    return saved_files

def get_file_stats():
    """Get statistics about processed files"""
    try:
        if os.path.exists("chunks.json"):
            with open("chunks.json", 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            file_types = {}
            file_count = {}
            
            for chunk in chunks:
                filename = chunk.get('filename', 'Unknown')
                file_type = chunk.get('file_type', 'Unknown')
                
                if file_type not in file_types:
                    file_types[file_type] = 0
                file_types[file_type] += 1
                
                if filename not in file_count:
                    file_count[filename] = 0
                file_count[filename] += 1
            
            return len(chunks), len(file_count), file_types
        return 0, 0, {}
    except:
        return 0, 0, {}

def auto_initialize_knowledge_base():
    """Automatically initialize knowledge base if data directory has files but knowledge base doesn't exist"""
    data_dir = "data"
    required_files = ["embeddings.npy", "faiss_index.index", "chunks.json"]
    
    # Check if knowledge base files exist
    kb_exists = all(os.path.exists(f) for f in required_files)
    
    # Check if data directory has files
    data_has_files = False
    if os.path.exists(data_dir):
        supported_extensions = ['.pdf', '.xlsx', '.xls', '.csv', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        files = [f for f in os.listdir(data_dir) if any(f.lower().endswith(ext) for ext in supported_extensions)]
        data_has_files = len(files) > 0
    
    # If data has files but knowledge base doesn't exist, create it
    if data_has_files and not kb_exists:
        st.info("Initializing knowledge base from existing files...")
        with st.spinner("Processing documents... This may take a few minutes."):
            try:
                from indexing import process_documents
                process_documents(data_dir, force_reprocess=True)
                st.success("Knowledge base initialized successfully!")
                st.session_state.knowledge_base_ready = True
                st.rerun()
            except Exception as e:
                st.error(f"Error initializing knowledge base: {e}")
                return False
    
    return kb_exists or data_has_files

def main():
    load_css()
    initialize_session_state()
    
    # Auto-initialize knowledge base if needed
    auto_initialize_knowledge_base()
    
    # Header
    logo_data = load_logo()
    
    if logo_data:
        # Header with logo
        st.markdown(f"""
        <div class="header-container">
            <div class="logo-container">
                <img src="{logo_data}" class="company-logo" alt="Company Logo">
                <h1 class="title">GESIL RAG System</h1>
            </div>
            <p class="subtitle">Intelligent Document Processing & Question Answering</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Header without logo (fallback)
        st.markdown("""
        <div class="header-container">
            <div class="logo-container">
                <h1 class="title">GESIL RAG System</h1>
            </div>
            <p class="subtitle">Intelligent Document Processing & Question Answering</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        # Knowledge base status
        st.subheader("Knowledge Base Status")
        if st.session_state.knowledge_base_ready:
            chunks_count, files_count, file_types = get_file_stats()
            st.success(f"Ready - {files_count} files, {chunks_count} chunks")
            
            if file_types:
                st.write("File Types:")
                for file_type, count in file_types.items():
                    st.write(f"  • {file_type}: {count} chunks")
        else:
            st.warning("No knowledge base found. Upload and process files first.")
        
        st.divider()
        
        # Actions
        st.subheader("Actions")
        
        if st.button("Refresh Knowledge Base", help="Check for new files and update"):
            with st.spinner("Refreshing..."):
                try:
                    process_documents()
                    st.session_state.knowledge_base_ready = check_knowledge_base()
                    st.success("Knowledge base refreshed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error refreshing: {e}")
        
        if st.button("Force Rebuild", help="Rebuild entire knowledge base from scratch"):
            with st.spinner("Clearing existing data and rebuilding..."):
                try:
                    # Clear existing knowledge base files first
                    files_to_clear = ["embeddings.npy", "faiss_index.index", "chunks.json"]
                    for file_path in files_to_clear:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    # Force rebuild
                    process_documents(force_reprocess=True)
                    st.session_state.knowledge_base_ready = check_knowledge_base()
                    
                    if st.session_state.knowledge_base_ready:
                        st.success("Knowledge base completely rebuilt from scratch!")
                    else:
                        st.warning("Rebuild completed but knowledge base files not found. Please ensure you have documents in the data directory.")
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error rebuilding: {e}")
                    # Try to restore knowledge base status
                    st.session_state.knowledge_base_ready = check_knowledge_base()
        
        st.divider()
        
        if st.button("🗑️ Clear All Data", help="Remove all processed data and uploaded files", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                with st.spinner("Clearing all data..."):
                    try:
                        # Clear knowledge base files
                        files_to_clear = ["embeddings.npy", "faiss_index.index", "chunks.json"]
                        for file_path in files_to_clear:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        
                        # Clear data directory
                        import shutil
                        if os.path.exists("data"):
                            shutil.rmtree("data")
                            os.makedirs("data")
                        
                        # Reset session state
                        st.session_state.knowledge_base_ready = False
                        st.session_state.processed_files = []
                        st.session_state.query_history = []
                        st.session_state.confirm_clear = False
                        
                        st.success("All data cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing data: {e}")
                        st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm - this will delete ALL data!")
                st.rerun()
        
        # Logo upload interface
        st.divider()
        upload_logo_interface()
    
    # Main content area with columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'xlsx', 'xls', 'csv', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Excel, CSV, Images"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"  • {file.name} ({file.size} bytes)")
            
            if st.button("Process Files", type="primary"):
                with st.spinner("Processing files..."):
                    try:
                        # Save uploaded files
                        saved_files = save_uploaded_files(uploaded_files)
                        
                        if saved_files:
                            st.success(f"Saved {len(saved_files)} files")
                            
                            # Process documents
                            process_documents()
                            st.session_state.knowledge_base_ready = check_knowledge_base()
                            
                            st.success("Files processed successfully!")
                            st.session_state.processed_files.extend(saved_files)
                            st.rerun()
                        else:
                            st.error("No files were saved")
                            
                    except Exception as e:
                        st.error(f"Error processing files: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        st.subheader("❓ Ask Questions")
        
        if st.session_state.knowledge_base_ready:
            # Load RAG components
            embeddings, index, chunks, embedding_model = load_rag_components()
            
            if all(component is not None for component in [embeddings, index, chunks, embedding_model]):
                query = st.text_area(
                    "Enter your question:",
                    height=100,
                    placeholder="Ask anything about your uploaded documents..."
                )
                
                if st.button("Get Answer", type="primary", disabled=not query.strip()):
                    if query.strip():
                        with st.spinner("Searching and generating answer..."):
                            try:
                                response = query_rag_system(
                                    query, index, embeddings, chunks, embedding_model
                                )
                                
                                # Store in history
                                st.session_state.query_history.append({
                                    'question': query,
                                    'answer': response
                                })
                                
                                # Display answer in response section
                                st.markdown('<div class="response-section">', unsafe_allow_html=True)
                                st.subheader("Answer:")
                                st.write(response)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error generating answer: {e}")
                    else:
                        st.warning("Please enter a question")
            else:
                st.error("Failed to load RAG components. Please refresh the knowledge base.")
        else:
            st.info("🔄 Upload documents to get started!")
            st.markdown("""
            **Getting Started:**
            1. Upload your documents (PDF, Excel, CSV, or Images)
            2. Click 'Process Files' to build the knowledge base
            3. Ask questions about your documents
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Query History
    if st.session_state.query_history:
        st.divider()
        st.subheader("📊 Recent Questions")
        
        # Show last 5 queries
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Q: {item['question'][:100]}..." if len(item['question']) > 100 else f"Q: {item['question']}"):
                st.write("**Answer:**")
                st.write(item['answer'])

if __name__ == "__main__":
    main()
