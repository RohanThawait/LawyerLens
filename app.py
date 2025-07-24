import streamlit as st
import pymupdf
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain_pinecone import Pinecone
import os

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
LLM_MODEL_NAME = "gemini-1.5-flash"

# Cache the embedding model loading
@st.cache_resource
def load_embeddings():
    """Loads the Hugging Face embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

# Cache the LLM loading
@st.cache_resource
def load_llm():
    """Loads the Google Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

def get_pdf_text(pdf_docs: List) -> str:
    """Extracts text from a list of uploaded PDF file objects."""
    text = ""
    for pdf_file in pdf_docs:
        try:
            with pymupdf.open(stream=pdf_file.read(), filetype="pdf") as doc:
                text += "".join(page.get_text() for page in doc)
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {e}")
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
    st.set_page_config(page_title="LawyerLens", page_icon="⚖️")
    st.header("LawyerLens: AI Legal Assistant ⚖️")

    # Load models and vector store using cached functions
    embeddings = load_embeddings()
    llm = load_llm()
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever_uploaded" not in st.session_state:
        st.session_state.retriever_uploaded = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your documents and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store_uploaded = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                    st.session_state.retriever_uploaded = vector_store_uploaded.as_retriever()
                    st.success("Documents processed!")
            else:
                st.warning("Please upload at least one PDF file.")

    # Main chat interface logic
    if user_prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            if st.session_state.retriever_uploaded is None:
                st.warning("Please upload and process a document first.")
                st.stop()

            with st.spinner("Analyzing..."):
                # 1. Connect to the Legal Knowledge Base in Pinecone
                embeddings = load_embeddings()
                vector_store_law = Pinecone.from_existing_index(
                    index_name="lawyerlens-kb",
                    embedding=embeddings,
                )
                retriever_indian_law = vector_store_law.as_retriever(search_kwargs={'k': 5})

                # 2. Retrieve context from BOTH sources
                docs_from_law = retriever_indian_law.get_relevant_documents(user_prompt)
                docs_from_upload = st.session_state.retriever_uploaded.get_relevant_documents(user_prompt)
                
                # Format retrieved docs into strings
                context_law = "\n".join([doc.page_content for doc in docs_from_law])
                context_upload = "\n".join([doc.page_content for doc in docs_from_upload])

                # 3. Define the prompt template for the comparison
                template = """
                    **ROLE:** You are an AI legal assistant with the expertise of a seasoned Indian High Court lawyer.

                    **TASK:** Provide a confident, accurate, and legally grounded answer to the user's question by comparing the 'Uploaded Document Context' with the 'Indian Law Context'.

                    **INSTRUCTIONS:**
                    - Get straight to the answer. Do not introduce yourself.
                    - If Indian Law Context is provided:
                    - Compare it directly with the Uploaded Document Context.
                    - If there is a conflict, explain the discrepancy clearly and state the correct legal position.
                    - If it aligns, confirm the document is compliant with Indian law.
                    - If Indian Law Context is silent on the issue, provide guidance based solely on the Uploaded Document Context.
                    - Do **not** include generic disclaimers like "consult a lawyer" or "get legal review".
                    - Avoid vague statements. Be specific and decisive in your analysis.

                    ---
                    **Indian Law Context:**
                    {context_indian_law}

                    **Uploaded Document Context:**
                    {context_uploaded_doc}

                    **User's Question:**
                    {question}
                    ---
                    **Legal Analysis:**
                    """

                prompt_template = PromptTemplate(
                    template=template, 
                    input_variables=["context_uploaded_doc", "context_indian_law", "question"]
                )

                # 4. Run the LLM Chain
                chain = LLMChain(llm=llm, prompt=prompt_template)
                response = chain.invoke({
                    "context_uploaded_doc": context_upload,
                    "context_indian_law": context_law,
                    "question": user_prompt
                })
                
                response_text = response['text']
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == '__main__':
    main()