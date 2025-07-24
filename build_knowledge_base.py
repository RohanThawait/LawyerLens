import os
import pymupdf
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from uuid import uuid4

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
SOURCE_DIRECTORY = "./legal_docs"

pinecone_api_key="PINECONE_API_KEY"
pc = Pinecone(api_key=pinecone_api_key)

index_name = "lawyerlens-kb"
index = pc.Index(index_name)
def get_pdf_text(pdf_paths: List[str]) -> str:
    """Extracts text from a list of PDF file paths."""
    text = ""
    for path in pdf_paths:
        try:
            with pymupdf.open(path) as doc:
                text += "".join(page.get_text() for page in doc)
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
    return text_splitter.split_text(raw_text)

def main():
    print("Building knowledge base and uploading to Pinecone...")

    # 1. Load source documents
    pdf_paths = [os.path.join(SOURCE_DIRECTORY, f) for f in os.listdir(SOURCE_DIRECTORY) if f.endswith(".pdf")]
    if not pdf_paths:
        print(f"No PDF documents found in the '{SOURCE_DIRECTORY}' folder.")
        return
    print(f"Found {len(pdf_paths)} documents to process.")

    # 2. Extract and chunk text
    raw_text = get_pdf_text(pdf_paths)
    chunks = get_text_chunks(raw_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"Split text into {len(documents)} chunks.")
    
    # 3. Initialize the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. Upload documents to Pinecone
    print(f"Uploading {len(documents)} chunks to Pinecone index 'lawyerlens-kb'...")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    print("\n Knowledge base uploaded to Pinecone successfully!")

if __name__ == '__main__':
    main()