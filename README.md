# ⚖️ LawyerLens: AI-Powered Legal Document Assistant

**[Live App Demo Link](https://huggingface.co/spaces/thawait/LawyerLens)**

LawyerLens is an advanced AI assistant designed to move beyond simple "chat with your PDF" functionality. Instead of just retrieving information, it validates clauses from uploaded legal documents (like rental agreements) against a curated, authoritative knowledge base of Indian law. This provides users with a powerful tool to check if their documents are aligned with legal statutes, offering a level of analysis that generic tools cannot.

![LawyerLens Demo GIF](https://github.com/RohanThawait/LawyerLens/raw/main/demo/Animation.gif)

---

## ## Key Features 🎯

* **Dual-Retrieval RAG Architecture:** For every user query, LawyerLens retrieves context from two sources simultaneously: the user's uploaded document and the permanent legal knowledge base.
* **Legal Validation:** The core feature. The AI is instructed to compare the two contexts and identify if the document's clause conflicts with, aligns with, or is silent on the matter according to the law.
* **Persistent Knowledge Base:** Built with **Pinecone** to store and efficiently search through a curated library of Indian legal texts.
* **Optimized & Local:** Uses powerful, open-source **Hugging Face** models for embeddings that run locally, ensuring privacy and cost-effectiveness. The final reasoning step is powered by the **Google Gemini API**.
* **Conversational Memory:** Remembers the context of the conversation for follow-up questions within a session.

---

## ## Architecture

The system uses a comparative RAG pipeline to deliver its analysis:

```
User Query -------------------------------------->+
                                                  |
+-----------------------------+     +-------------------------------+     +--------------------+
|   Temporary Vector Store    |     |   Cloud Vector Database       |     |      Google        |
| (Uploaded Doc - FAISS)      |     |     (Indian Law - Pinecone)   |     |      Gemini        |
+-----------------------------+     +-------------------------------+     +--------------------+
            |                                     |                               ^
            | (Retrieval 1)                       | (Retrieval 2)                 | (LLM Chain)
            v                                     v                               |
+-----------------------------+     +-------------------------------+             |
|   Context from Uploaded Doc |     |  Context from Indian Law      |-------------+
+-----------------------------+     +-------------------------------+
```

---

## ## Tech Stack 🛠️

* **Application Framework:** Streamlit
* **LLM & Orchestration:** LangChain, Google Gemini API
* **Embeddings:** Hugging Face Sentence Transformers (`hkunlp/instructor-large`)
* **Vector Databases:** Pinecone (for persistent knowledge base), FAISS (for temporary session documents)
* **Core Libraries:** PyMuPDF
* **Language:** Python

---

## ## Setup & Installation

Follow these steps to set up and run the project locally.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/LawyerLens.git](https://github.com/your-username/LawyerLens.git)
cd LawyerLens
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Set up API Keys:**
Create a file at `.streamlit/secrets.toml` and add your Google API key:
```toml
GOOGLE_API_KEY="AI..."
PINECONE_API_KEY = "..."
PINECONE_ENVIRONMENT = "..."
```

**4. Build the Knowledge Base:**
Place your source legal PDFs (e.g., "The Indian Contract Act, 1872.pdf") into the `legal_docs` folder. Then, run the builder script once:
```bash
python build_knowledge_base.py
```
This will populate your cloud-hosted Pinecone index with the embedded legal documents.

**5. Run the Streamlit App:**
```bash
streamlit run app.py
```
Open your browser to the local URL provided by Streamlit.
