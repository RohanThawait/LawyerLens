# app.py
import streamlit as st
import pymupdf 
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# We will add more imports here as we build

# --- Function to extract text from PDF ---
def get_pdf_text(pdf_docs: List) -> str:
    """
    Extracts text from a list of uploaded PDF file objects.
    """
    text = ""
    for pdf_file in pdf_docs:
        try:
            # Open the PDF file from the in-memory stream
            with pymupdf.open(stream=pdf_file.read(), filetype="pdf") as doc:
                text += "".join(page.get_text() for page in doc)
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {e}")
    # Placeholder for PDF text extraction logic
    st.write("Extracting text... (This part we will build first!)")
    return text

# --- Function to split text into chunks ---
def get_text_chunks(raw_text):
    """
    Splits a large string of text into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# --- Function to create and save vector store ---
def get_vector_store(text_chunks: List[str]):
    """
    Creates a FAISS vector store from text chunks using a local embedding model.
    """
    # The model name for our chosen sentence transformer
    model_name = "hkunlp/instructor-large" # A good alternative for legal text
    # or use the one you mentioned: "sentence-transformers/all-MiniLM-L6-v2"

    # Initialize the embeddings using the local model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create the vector store
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(page_title="LawyerLens", page_icon="⚖️")
    st.header("LawyerLens: AI Legal Assistant ⚖️")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing documents..."):
                # 1. Get PDF Text
                raw_text = get_pdf_text(pdf_docs)
                
                # 2. Get Text Chunks
                text_chunks = get_text_chunks(raw_text)
                
                # 3. Create Vector Store with local embeddings
                vector_store = get_vector_store(text_chunks)
                st.success("Documents processed!")

                # 4. Create Conversation Chain
                # This chain uses an LLM for chat and our vector_store for retrieval
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(),
                    memory=memory
                )
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        # Display conversation
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"**You:** {message.content}")
            else:
                st.write(f"**LawyerLens:** {message.content}")

# To run the app
if __name__ == '__main__':
    main()