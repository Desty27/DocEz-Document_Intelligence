import os
import pypdf
import pandas as pd
import easyocr
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from datetime import datetime

# 1. Load and process documents from the 'data' directory
def load_documents(data_dir='data'):
    documents = []
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist. Creating it...")
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created directory '{data_dir}'. Please add files to this directory and run the script again.")
        return documents
    
    # Get all supported files, but filter out problematic ones
    supported_extensions = ['.pdf', '.xlsx', '.xls', '.csv', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    all_files = os.listdir(data_dir)
    files = []
    
    for f in all_files:
        if any(f.lower().endswith(ext) for ext in supported_extensions):
            # Check if filename has problematic characters but don't skip completely
            # Instead, we'll handle them in the processing functions
            files.append(f)
    
    if not files:
        print(f"No supported files found in '{data_dir}' directory. Supported formats: PDF, Excel, CSV, Images")
        return documents
    
    # Initialize OCR reader
    try:
        ocr_reader = easyocr.Reader(['en'])
    except:
        print("Warning: EasyOCR not available. Image processing will be skipped.")
        ocr_reader = None
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        try:
            content = ""
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.pdf':
                content = process_pdf(filepath)
            elif file_ext in ['.xlsx', '.xls']:
                content = process_excel(filepath)
            elif file_ext == '.csv':
                content = process_csv(filepath)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'] and ocr_reader:
                content = process_image(filepath, ocr_reader)
            
            if content:
                documents.append({
                    "filename": filename, 
                    "content": content,
                    "file_type": file_ext,
                    "processed_date": datetime.now().isoformat()
                })
                print(f"Loaded: {filename} ({file_ext})")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return documents

def process_pdf(filepath):
    """Extract text from PDF files"""
    with open(filepath, 'rb') as f:
        reader = pypdf.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def process_excel(filepath):
    """Extract text from Excel files"""
    try:
        # Read all sheets
        xl_file = pd.ExcelFile(filepath)
        content = ""
        filename = os.path.basename(filepath)
        
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            
            # Clean the dataframe
            df = df.dropna(how='all')  # Remove completely empty rows
            df = df.fillna('')  # Fill NaN with empty string
            
            content += f"=== Excel File: {filename} - Sheet: {sheet_name} ===\n"
            
            # Add column headers context
            if not df.empty:
                content += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
                
                # Convert to string representation with better formatting
                for idx, row in df.iterrows():
                    row_text = []
                    for col, val in row.items():
                        if str(val).strip():  # Only include non-empty values
                            row_text.append(f"{col}: {val}")
                    
                    if row_text:  # Only add if there's actual content
                        content += " | ".join(row_text) + "\n"
                
                content += "\n" + "="*50 + "\n\n"
            else:
                content += "No data found in this sheet.\n\n"
        
        return content
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        return f"Error reading Excel file {os.path.basename(filepath)}: {str(e)}"

def process_csv(filepath):
    """Extract text from CSV files"""
    try:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        
        # Clean the dataframe
        df = df.dropna(how='all')  # Remove completely empty rows
        df = df.fillna('')  # Fill NaN with empty string
        
        content = f"=== CSV File: {filename} ===\n"
        
        if not df.empty:
            content += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
            
            # Convert to string representation with better formatting
            for idx, row in df.iterrows():
                row_text = []
                for col, val in row.items():
                    if str(val).strip():  # Only include non-empty values
                        row_text.append(f"{col}: {val}")
                
                if row_text:  # Only add if there's actual content
                    content += " | ".join(row_text) + "\n"
            
            content += "\n" + "="*50 + "\n"
        else:
            content += "No data found in this CSV file.\n"
        
        return content
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return f"Error reading CSV file {os.path.basename(filepath)}: {str(e)}"

def process_image(filepath, ocr_reader):
    """Extract text from images using OCR"""
    try:
        # Check if file exists and is readable
        if not os.path.exists(filepath):
            return "Image file not found"
        
        filename = os.path.basename(filepath)
        print(f"Processing image: {filename}")
        
        # First try to read with PIL to ensure it's a valid image
        try:
            img = Image.open(filepath)
            img.verify()  # Verify it's a valid image
        except Exception as pil_error:
            print(f"PIL verification failed: {pil_error}")
            return f"Invalid image format: {str(pil_error)}"
        
        # Reopen the image after verify (verify closes the file)
        try:
            img = Image.open(filepath)
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Convert PIL image to numpy array for EasyOCR
            import numpy as np
            img_array = np.array(img)
            
            # Check if image array is valid
            if img_array is None or img_array.size == 0:
                return "Invalid image data"
            
            print(f"Image shape: {img_array.shape}")
            
            # For problematic filenames, save a temporary clean copy
            temp_image_path = None
            if any(char in filename for char in ['Γ', 'ï', 'â', '_', ' ']):
                try:
                    import tempfile
                    import shutil
                    
                    # Create a temporary file with a clean name
                    temp_dir = tempfile.gettempdir()
                    clean_filename = f"temp_ocr_{hash(filepath) % 10000}.png"
                    temp_image_path = os.path.join(temp_dir, clean_filename)
                    
                    # Save the image to the temporary location
                    img.save(temp_image_path, 'PNG')
                    print(f"Created temporary image: {temp_image_path}")
                    
                    # Use the temporary file for OCR
                    result = ocr_reader.readtext(temp_image_path)
                except Exception as temp_error:
                    print(f"Temporary file approach failed: {temp_error}")
                    # Fall back to numpy array approach
                    result = ocr_reader.readtext(img_array)
                finally:
                    # Clean up temporary file
                    if temp_image_path and os.path.exists(temp_image_path):
                        try:
                            os.remove(temp_image_path)
                        except:
                            pass
            else:
                # Use OCR directly on the numpy array
                result = ocr_reader.readtext(img_array)
            
            if result and len(result) > 0:
                # Extract text from OCR results
                extracted_texts = []
                for item in result:
                    if len(item) >= 2 and isinstance(item[1], str) and item[1].strip():
                        # Also include confidence score if available
                        confidence = item[2] if len(item) > 2 else 1.0
                        if confidence > 0.5:  # Only include text with reasonable confidence
                            extracted_texts.append(item[1].strip())
                
                if extracted_texts:
                    text = " ".join(extracted_texts)
                    print(f"Extracted text ({len(extracted_texts)} segments, {len(text)} characters)")
                    return text
                else:
                    return "No readable text found in image (low confidence)"
            else:
                return "No text detected in image"
                
        except Exception as ocr_error:
            print(f"OCR processing error: {ocr_error}")
            return f"OCR processing failed: {str(ocr_error)}"
            
    except Exception as e:
        print(f"Error processing image file: {e}")
        return f"Error reading image: {str(e)}"

# 2. Chunk documents (simple chunking by page for now, can be improved)
def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    chunks = []
    for doc in documents:
        content = doc["content"]
        filename = doc["filename"]
        # Simple chunking by sliding window
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            chunks.append({"filename": filename, "chunk": chunk})
    return chunks

# 3. Create embeddings using sentence-transformers model
def create_embeddings(chunks):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode([chunk["chunk"] for chunk in chunks])
    return embeddings

# 4. Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(np.float32(embeddings))
    return index

# 5. Load existing data for persistent memory
def load_existing_data():
    """Load existing chunks and embeddings if they exist"""
    existing_chunks = []
    existing_embeddings = None
    
    if os.path.exists("chunks.json"):
        with open("chunks.json", 'r', encoding='utf-8') as f:
            existing_chunks = json.load(f)
        print(f"Loaded {len(existing_chunks)} existing chunks")
    
    if os.path.exists("embeddings.npy"):
        existing_embeddings = np.load("embeddings.npy")
        print(f"Loaded existing embeddings with shape: {existing_embeddings.shape}")
    
    return existing_chunks, existing_embeddings

# 6. Merge new and existing data
def merge_data(existing_chunks, existing_embeddings, new_chunks, new_embeddings):
    """Merge existing and new data, avoiding duplicates"""
    # If no existing data, just return new data
    if not existing_chunks or existing_embeddings is None:
        print(f"No existing data, using {len(new_chunks)} new chunks")
        return new_chunks, np.array(new_embeddings) if len(new_embeddings) > 0 else None
    
    # Check for duplicates based on filename and chunk content
    existing_files = set()
    for chunk in existing_chunks:
        existing_files.add(chunk.get('filename', ''))
    
    # Filter out chunks from files that already exist
    filtered_new_chunks = []
    filtered_new_embeddings = []
    
    for i, chunk in enumerate(new_chunks):
        if chunk.get('filename', '') not in existing_files:
            filtered_new_chunks.append(chunk)
            filtered_new_embeddings.append(new_embeddings[i])
    
    if not filtered_new_chunks:
        print("No new files to add (all files already processed)")
        return existing_chunks, existing_embeddings
    
    # Combine data
    all_chunks = existing_chunks + filtered_new_chunks
    
    if len(filtered_new_embeddings) > 0:
        all_embeddings = np.vstack([existing_embeddings, np.array(filtered_new_embeddings)])
    else:
        all_embeddings = existing_embeddings
    
    print(f"Added {len(filtered_new_chunks)} new chunks")
    return all_chunks, all_embeddings

def process_documents(data_dir='data', force_reprocess=False):
    """Main function to process documents with persistent memory"""
    if force_reprocess:
        print("Force reprocessing all documents...")
        # Clear existing files
        files_to_clear = ["embeddings.npy", "faiss_index.index", "chunks.json"]
        for file_path in files_to_clear:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Cleared existing {file_path}")
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
        
        existing_chunks, existing_embeddings = [], None
    else:
        existing_chunks, existing_embeddings = load_existing_data()
    
    # Load new documents
    documents = load_documents(data_dir)
    
    if not documents:
        if existing_chunks and not force_reprocess:
            print("No new documents found, but existing data is available.")
            return existing_chunks, existing_embeddings
        else:
            print("No documents found.")
            return [], None
    
    # Process new documents
    new_chunks = chunk_documents(documents)
    new_embeddings = create_embeddings(new_chunks)
    
    # Merge with existing data (will be empty if force_reprocess=True)
    all_chunks, all_embeddings = merge_data(existing_chunks, existing_embeddings, new_chunks, new_embeddings)
    
    if all_embeddings is not None:
        # Build FAISS index
        index = build_faiss_index(all_embeddings)
        
        # Save everything
        np.save("embeddings.npy", all_embeddings)
        faiss.write_index(index, "faiss_index.index")
        
        with open("chunks.json", 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Embeddings shape: {all_embeddings.shape}")
        print("Files saved: embeddings.npy, faiss_index.index, chunks.json")
        
        return all_chunks, all_embeddings
    
    return [], None

if __name__ == '__main__':
    import sys
    
    # Check for force reprocess flag
    force_reprocess = '--force' in sys.argv or '-f' in sys.argv
    
    if force_reprocess:
        print("Force reprocessing enabled - will rebuild entire knowledge base")
    
    chunks, embeddings = process_documents(force_reprocess=force_reprocess)
    
    if chunks and embeddings is not None:
        print(f"\nKnowledge base ready with {len(chunks)} chunks!")
    else:
        print("\nNo data processed. Please add supported files to the 'data' directory.")
        print("Supported formats: PDF, Excel (.xlsx, .xls), CSV, Images (.png, .jpg, .jpeg, .tiff, .bmp)")