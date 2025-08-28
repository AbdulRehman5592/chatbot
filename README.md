# Multi-PDF Chatbot with OCR, RAG, and Web Search

## Overview

This project is an intelligent assistant that allows users to chat with the content of multiple PDF documents. It leverages OCR (Optical Character Recognition) to extract text from PDFs, builds a vector store for semantic search, and uses Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM) to answer user queries. If the answer is not found in the local documents, the system can automatically perform a web search and use those results to answer the query.

The system consists of a FastAPI backend, a Streamlit frontend, and several utility modules for PDF processing, vector storage, and performance monitoring.

---

## Main Components

### 1. Backend (`backend.py`)

- **Framework:** FastAPI
- **Key Endpoints:**
  - `/upload_pdfs/` and `/upload_pdfs_base64/`: Accept PDF files, extract text and images, perform OCR, and build a vector store for semantic search.
  - `/chat/`: Accepts user queries, retrieves relevant document chunks, and generates answers using an LLM. If the answer is not found locally, it triggers a web search.
  - `/reset/`, `/history/`, `/performance_metrics/`: Session management and monitoring.
  - `/health`: Health check endpoint (including Tesseract OCR availability).

- **Performance Monitoring:** Each major operation (PDF processing, vector store creation, similarity search, LLM inference) is timed and logged.

---

### 2. Frontend (`streamlit_app.py`)

- **Framework:** Streamlit
- **Features:**
  - File upload interface for PDFs.
  - Chat interface for user queries.
  - Displays conversation history and highlights sources (including bounding boxes for images).
  - Visualizes the workflow using Mermaid diagrams.
  - Handles session state and displays performance metrics.

---

### 3. PDF and OCR Processing (`pdf_utils.py`, `PDF_Processing_and_OCR_Implementation.ipynb`)

- **PDF Extraction:** Uses PyMuPDF to extract text and images from PDFs.
- **OCR:** Uses Tesseract (via pytesseract) to extract text from images within PDFs.
- **Chunking:** Splits extracted text into manageable chunks for semantic search.

---

### 4. Vector Store (`vectorstore_utils.py`)

- **Embedding:** Uses Google Generative AI Embeddings to convert text chunks into vectors.
- **Storage:** Uses FAISS for efficient similarity search over document chunks.

---

### 5. LLM Integration and RAG (`llm_utils.py`, `langgraph_workflow.py`)

- **LLM:** Uses Gemini (Google Generative AI) via LangChain.
- **Prompt Engineering:** Custom prompts guide the LLM to use both conversation history and document context.
- **RAG Workflow:** Orchestrated using LangGraph, which manages the flow: vector store loading → similarity search → LLM inference → (if needed) web search.
- **Web Search:** If the answer is not found in local documents, the system uses Tavily to search the web and passes those results to the LLM.

---

### 6. Session and History Management (`history.py`)

- **Tracks:** User queries, answers, sources, and timestamps for each session.

---

### 7. Configuration and Environment

- **API Keys:** Managed via `.env` and `config.py`.
- **Dependencies:** Listed in `requirements.txt`.

---

## How It Works

1. **User uploads PDFs** via the Streamlit frontend.
2. **Backend extracts text and images** from PDFs, performs OCR, and builds a vector store.
3. **User submits a query** through the chat interface.
4. **Backend searches for relevant chunks** in the vector store and passes them, along with conversation history, to the LLM.
5. **LLM generates an answer** using both the document context and conversation history.
6. **If the answer is not found locally,** the system performs a web search and uses those results to answer the query.
7. **Sources and bounding boxes** are returned for local document answers; for web answers, only URLs are cited.

---

## Key Technologies

- **FastAPI**: Backend API
- **Streamlit**: Frontend UI
- **PyMuPDF, pytesseract**: PDF and OCR processing
- **FAISS**: Vector similarity search
- **LangChain, Gemini (Google Generative AI)**: LLM and RAG
- **Tavily**: Web search API
- **Mermaid**: Workflow visualization

---

## Setup and Running

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Set up environment variables** in `.env` (API keys for Google, Tavily, etc.).
3. **Run the backend:**
   ```
   python backend.py
   ```
4. **Run the frontend:**
   ```
   streamlit run streamlit_app.py
   ```

---

## Notes

- The system is designed to be extensible for other document types and LLM providers.
- Performance metrics are tracked and can be saved for analysis.
- The workflow is visualized for easier debugging and understanding. 