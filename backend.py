import time
import streamlit as st
import requests as requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from dotenv import load_dotenv
from uuid import uuid4
import base64
import os

from config import GOOGLE_API_KEY
from history import save_history, get_history, clear_history
from pdf_utils import extract_text_from_pdfs
from vectorstore_utils import chunk_text, create_vector_store, load_vector_store
from llm_utils import get_chain 
from performance_monitor import PerformanceMonitor
from langgraph_workflow import run_chat_workflow_async, ChatState

from ocr_txt_search_utils import answer_from_txt_files


import logging


# Setup for logging
logging.basicConfig(level=logging.INFO)

performance_monitor = PerformanceMonitor()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload_pdfs/")
@performance_monitor.timing_decorator("upload_pdfs_endpoint")
async def upload_pdfs(files: list[UploadFile] = File(...), session_id: str = Form(None)):
    if not session_id or session_id == "string":
        session_id = str(uuid4())
    start_time = time.time()
    ocr_text, _, pdf_names, ocr_chunks, ocr_metadatas = await extract_text_from_pdfs(files, session_id=session_id)
    pdf_processing_time = (time.time() - start_time) * 1000
    performance_monitor.metrics["pdf_processing"].append(pdf_processing_time)
    print(f"[PERFORMANCE] PDF Processing: {pdf_processing_time:.2f}ms")
    if not ocr_chunks:
        return JSONResponse(status_code=400, content={"error": "No text extracted from PDFs."})
    # Time vector store creation
    start_time = time.time()
    create_vector_store(ocr_chunks, session_id, ocr_metadatas)
    vector_store_time = (time.time() - start_time) * 1000
    performance_monitor.metrics["vector_store_creation"].append(vector_store_time)
    print(f"[PERFORMANCE] Vector Store Creation: {vector_store_time:.2f}ms")
    return {
        "session_id": session_id, 
        "pdf_names": pdf_names, 
        "chunks": len(ocr_chunks),
        'OCR': ocr_text,
        "performance_metrics": {
            "pdf_processing_ms": pdf_processing_time,
            "vector_store_creation_ms": vector_store_time
        }
    }

@app.post("/upload_pdfs_base64/")
async def upload_pdfs_base64(
    files_base64: list[str] = Body(...),
    filenames: list[str] = Body(...),
    session_id: str = Form(None)
):
    if not session_id or session_id == "string":
        session_id = str(uuid4())
    temp_files = []
    try:
        for b64, fname in zip(files_base64, filenames):
            pdf_bytes = base64.b64decode(b64)
            temp_path = f"pdf_output/{session_id or 'default'}/{fname}"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)
            class DummyUploadFile:
                def __init__(self, path, filename):
                    self.file = open(path, "rb")
                    self.filename = filename
                async def read(self):
                    self.file.seek(0)
                    return self.file.read()
            temp_files.append(DummyUploadFile(temp_path, fname))
        start_time = time.time()
        ocr_text, _, pdf_names, ocr_chunks, ocr_metadatas = await extract_text_from_pdfs(temp_files, session_id=session_id)
        pdf_processing_time = (time.time() - start_time) * 1000
        performance_monitor.metrics["pdf_processing"].append(pdf_processing_time)
        print(f"[PERFORMANCE] PDF Processing: {pdf_processing_time:.2f}ms")
        for f in temp_files:
            f.file.close()
        if not ocr_chunks:
            return JSONResponse(status_code=400, content={"error": "No text extracted from PDFs."})
        start_time = time.time()
        create_vector_store(ocr_chunks, session_id, ocr_metadatas)
        vector_store_time = (time.time() - start_time) * 1000
        performance_monitor.metrics["vector_store_creation"].append(vector_store_time)
        print(f"[PERFORMANCE] Vector Store Creation: {vector_store_time:.2f}ms")
        return {
            "session_id": session_id,
            "pdf_names": pdf_names,
            "chunks": len(ocr_chunks),
            'OCR': ocr_text,
            "performance_metrics": {
                "pdf_processing_ms": pdf_processing_time,
                "vector_store_creation_ms": vector_store_time
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat/")
@performance_monitor.timing_decorator("chat_endpoint")
async def chat(query: str = Form(...), session_id: str = Form(...)):
    if not session_id:
        return JSONResponse(status_code=400, content={"error": "session_id is required."})
    
    try:
        # Use LangGraph workflow from langgraph_workflow.py
        conversation_history = get_history(session_id)
        state: ChatState = {
            "query": query, 
            "session_id": session_id, 
            "answer": "", 
            "performance": {}, 
            "messages": conversation_history.messages if hasattr(conversation_history, 'messages') else conversation_history
        }
        
        # Invoke the workflow
        result = await run_chat_workflow_async(state)
        
        # Extract results and metadata
        answer = result.get("answer", "No answer could be generated.")
        timestamp = result.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # --- Extract bounding boxes from docs metadata ---
        docs = result.get("docs", [])
        bboxes = []
        for doc in docs:
            source_val = doc.metadata.get('source')
            bbox_val = doc.metadata.get('bbox')
            if source_val and bbox_val:
                bboxes.append({
                    "source": source_val,
                    "bbox": bbox_val
                })

        # --- Add sources to the answer ---
        sources = []
        for i, doc in enumerate(docs):
            src = doc.metadata.get('source')
            bbox = doc.metadata.get('bbox')
            if src and src not in ('selectable_text', 'Unknown'):
                if bbox:
                    sources.append(f"Source:[File: {src}, BBox: {bbox}]")
                else:
                    sources.append(f"Source:[File: {src}]")
        source_str = " ".join(sources)
        final_answer_with_source = f"{answer}\n\n{source_str}" if sources else answer

        # --- Return response with bboxes ---
        return {
            "answer": final_answer_with_source, 
            "session_id": session_id, 
            "bboxes": bboxes,  # Always return bboxes for highlighting
            "timestamp": timestamp
        }

    except Exception as e:
        logging.error(f"Error during chat: {e}")
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred: {str(e)}"})

@app.post("/reset/")
async def reset_session(session_id: str = Form(...)):
    clear_history(session_id)
    # Optionally, remove FAISS index for session
    import shutil
    try:
        shutil.rmtree(f"faiss_index/{session_id}")
    except Exception:
        pass
    return {"status": "reset", "session_id": session_id}

@app.get("/history/")
async def history(session_id: str):
    history = get_history(session_id)
    return {"session_id": session_id, "history": history}

@app.get("/performance_metrics/")
async def get_performance_metrics():
    """Get aggregated performance metrics"""
    return performance_monitor.get_metrics_summary()

@app.get("/performance_metrics/save/")
async def save_performance_metrics():
    """Save current metrics to file"""
    performance_monitor.save_metrics_to_file()
    return {"message": "Metrics saved successfully"}

@app.get("/")
def root():
    return {"status": "FastAPI backend running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Tesseract
        import pytesseract
        version = pytesseract.get_tesseract_version()
        return {
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "tesseract_version": str(version)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Add this at the end of the file
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",  # Important: Use 0.0.0.0 to bind to all interfaces
        port=port,
        reload=False  # Set to False for production
    )


