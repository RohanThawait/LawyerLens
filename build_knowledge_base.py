# build_knowledge_base.py

import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pymupdf 
from typing import List

# --- Reusable Functions from our main app ---
def get_pdf_text(pdf_paths: List[str]) -> str:
    """Extracts text from a list of PDF file paths."""
    text = ""
    for path in pdf_paths:
        try:
            with pymupdf.open(path) as doc:
                for page in doc:
                    text += page.get_text() or ""
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return text

def get_text_chunks(raw_text: str) -> List[str]:
    """Splits a large string of text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# --- Main Script Logic ---
def main():
    print("Building knowledge base...")

    # Define paths
    pdf_folder_path = "./legal_docs"
    chroma_db_path = "./chroma_db"

    # 1. Get list of PDF file paths
    pdf_paths = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]
    
    if not pdf_paths:
        print("No PDF documents found in the 'legal_docs' folder.")
        return

    # 2. Extract and Chunk Text
    raw_text = get_pdf_text(pdf_paths)
    chunks = get_text_chunks(raw_text)
    
    # 3. Initialize Embeddings and ChromaDB
    # Using a local, open-source model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
    
    # This creates a persistent database in the specified directory
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    # Create or get the collection (like a table in a regular database)
    collection = client.get_or_create_collection(name="indian_law")
    
    # 4. Add documents to the collection
    # ChromaDB handles the embedding process automatically here.
    # We add the chunks in batches to be efficient.
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        # Create unique IDs for each chunk in the batch
        start_id = collection.count()
        ids = [f"chunk_{j}" for j in range(start_id, start_id + len(batch_chunks))]
        
        collection.add(
            documents=batch_chunks,
            ids=ids
        )
        print(f"Added batch {i//batch_size + 1} to the collection.")

    print("\n✅ Knowledge base built successfully!")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == '__main__':
    main()