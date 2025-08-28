import streamlit as st
import requests
import shutil
import os
from dotenv import load_dotenv
#from langgraph_workflow import visualize_workflow_mermaid
#from streamlit_mermaid import st_mermaid
import re
#import speech_recognition as sr
from PIL import Image, ImageDraw

# Initialize session state variables at the very top
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Get backend URL from environment variable or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/")


def display_source_file(source_files_str, session_id, bboxes=None):
    """Displays source files (images or PDFs) and highlights bounding boxes on images."""
    if not session_id:
        st.warning("Cannot display source file: session_id is missing.")
        return

    # Extract all [File: ..., BBox: ...] entries from the answer string
    source_entries = re.findall(r"\[Source:\s*([^,\]]+),\s*BBox:\s*(\[[^\]]+\])\]", source_files_str)
    st.markdown(f"**Source Files:** {source_entries}")
    if not source_entries:
        st.warning(f"Could not extract valid source file from: {source_files_str}")
        print(f"[DEBUG] Source extraction failed for: {source_files_str}")
        return

    # Create a mapping from source file to its bounding boxes
    bbox_map = {}
    if bboxes:
        for entry in bboxes:
            source_file = entry.get('source')
            if source_file not in bbox_map:
                bbox_map[source_file] = []
            bbox_map[source_file].append(entry)
            st.markdown(f"**Boxes:** {bbox_map[source_file]}")

    for source_file, bbox_str in source_entries:
        if not source_file or source_file == 'N/A':
            continue
        img_path_pdf_output = os.path.join(f"pdf_output/{session_id}", source_file)
        file_displayed = False
        image_extensions = (".jpeg", ".jpg", ".png", ".gif", ".bmp")
        if source_file.lower().endswith(image_extensions):
            if os.path.exists(img_path_pdf_output):
                image = Image.open(img_path_pdf_output)
                # Draw the bbox from the answer string (not from bboxes list)
                try:
                    bbox = eval(bbox_str)
                    if bbox and len(bbox) == 4:
                        draw = ImageDraw.Draw(image)
                        draw.rectangle(bbox, outline="red", width=2)
                except Exception as e:
                    print(f"[DEBUG] Failed to parse bbox: {bbox_str} ({e})")
                st.image(image, caption=f"Source: {source_file}")
                file_displayed = True
        elif source_file.lower().endswith(".pdf"):
            pdf_path_session = os.path.join("pdf_output", session_id, source_file)
            if os.path.exists(pdf_path_session):
                st.markdown(f'<iframe src="/{pdf_path_session}" width="700" height="500"></iframe>', unsafe_allow_html=True)
                file_displayed = True
        if not file_displayed and "http" not in source_file.lower():
            st.warning(f"Source file '{source_file}' not found.")


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")
    # Remove: audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "webm"])
    
    # Display conversation history using chat messages
    if st.session_state.conversation_history:
        st.write("### Conversation History")
        for entry in st.session_state.conversation_history:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(entry["question"])
            
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(entry["answer"])
                # Check for and display source file with bboxes
                if "http"not in entry["answer"]:
                    bboxes = entry.get("bboxes")
                    context_type="local"
                    display_source_file(entry["answer"], st.session_state.session_id, bboxes=bboxes)

    # Use a session state variable for user_question if set by speech
    if st.session_state.get("clear_user_question", False):
        st.session_state['user_question'] = ""
        st.session_state['clear_user_question'] = False
    if st.session_state.get("clear_speech_text", False):
        st.session_state['speech_text'] = ""
        st.session_state['clear_speech_text'] = False
    # Copy speech to question if flag is set
    if st.session_state.get("copy_speech_to_question", False):
        st.session_state['user_question'] = st.session_state.get('speech_text', "")
        st.session_state['copy_speech_to_question'] = False
    user_question = st.session_state.get('user_question', "")
    user_question = st.text_input("Ask a Question from the PDF Files", value=user_question, key="user_question")
    if st.button("Ask"):
        if not user_question:
            st.warning("Please enter a question.")
        elif not st.session_state.session_id:
            st.warning("Please upload and process PDFs first.")
        else:
            with st.spinner("Getting answer..."):
                data = {"query": user_question, "session_id": st.session_state.session_id}
                response = requests.post(f"{BACKEND_URL}/chat/", data=data)
                if response.status_code == 200:
                    result = response.json()
                    print(f"**Result:** {result['answer']}")
                    st.session_state.conversation_history.append({
                        "question": user_question,
                        "answer": result["answer"],
                        "timestamp": result.get("timestamp", ""),
                        "bboxes": result.get("bboxes") # Store bboxes
                    })
                    # Set flags to clear prompt and speech boxes after answer
                    st.session_state['clear_user_question'] = True
                    st.session_state['clear_speech_text'] = True
                    st.rerun() # Rerun to display the new message and full history
                else:
                    st.error(response.json().get("error", "Failed to get answer."))

    # --- Speech Recognition Section (moved below prompt box and answer) ---
    st.markdown("#### 🎤 Speech to Text (Google Speech Recognition)")
    if 'speech_text' not in st.session_state:
        st.session_state.speech_text = ""
    if st.button("Record from Microphone"):
        if not st.session_state.session_id:
            st.warning("Please upload and process PDFs first.")
        else:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Please speak now...")
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    st.info("Transcribing...")
                    text = recognizer.recognize_google(audio)
                    st.session_state.speech_text = text
                    st.success("Recognized: " + text)
                except sr.WaitTimeoutError:
                    st.error("Listening timed out while waiting for phrase to start.")
                except sr.UnknownValueError:
                    st.error("Google Speech Recognition could not understand audio.")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")
    # Show recognized text and allow to copy to question box
    if st.session_state.speech_text:
        st.text_area("Recognized Speech", st.session_state.speech_text, height=80)
        if st.button("Copy to Question Box"):
            st.session_state['copy_speech_to_question'] = True
            st.rerun()
    # --- End Speech Recognition Section ---
    
    st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    @keyframes slideInFromLeft {
        0% {
            transform: translateX(-100%);
            opacity: 0;
        }
        70% {
            transform: translateX(10px);
            opacity: 1;
        }
        100% {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Main app background - Claude's warm cream/beige */
    .stApp {
        background: linear-gradient(135deg, #faf8f5 0%, #f5f2ed 100%);
        background-attachment: fixed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #2c2c2c;
    }
    
    /* Header styling - clean and minimal like Claude */
    header[data-testid="stHeader"] {
        background: rgba(250, 248, 245, 0.95) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        height: 60px !important;
        transition: all 0.3s ease;
    }
    
    /* Sidebar styling - clean white like Claude's sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.08);
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: none;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.05);
        animation: slideInFromLeft 0.8s ease-out;
        padding: 20px 16px;
    }
    
    /* Sidebar text styling */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stButton,
    section[data-testid="stSidebar"] label {
        color: #2c2c2c !important;
    }
    
    /* Sidebar badge styling for LinkedIn/GitHub */
    section[data-testid="stSidebar"] .stMarkdown img {
        margin: 2px;
        border-radius: 6px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    section[data-testid="stSidebar"] .stMarkdown img:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 123, 181, 0.3);
    }
    
    /* Sidebar title styling */
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1a1a1a !important;
        font-weight: 600;
        margin-bottom: 16px;
        margin-top: 20px;
        font-size: 18px;
        border-bottom: 2px solid rgba(0, 0, 0, 0.1);
        padding-bottom: 8px;
    }
    
    /* Sidebar radio button styling */
    section[data-testid="stSidebar"] .stRadio {
        margin-bottom: 20px;
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        color: #2c2c2c !important;
        font-weight: 500;
        font-size: 14px;
    }
    
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        background: rgba(0, 0, 0, 0.02);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    
    section[data-testid="stSidebar"] .stRadio [role="radio"] {
        background: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] .stRadio [role="radio"]:hover {
        background: rgba(0, 0, 0, 0.05);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    section[data-testid="stSidebar"] .stRadio [aria-checked="true"] {
        background: rgba(59, 130, 246, 0.1) !important;
        border-color: #3b82f6 !important;
        color: #2c2c2c !important;
    }
    
    /* Sidebar button styling */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #2c2c2c;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        width: 100%;
        margin: 4px 0;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        border-color: rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        color: #2c2c2c;  
    }
    
    /* Special styling for Reset and Rerun buttons */
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-color: rgba(220, 38, 38, 0.2);
        color: #dc2626;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15);
    }
    
    /* File uploader in sidebar */
    section[data-testid="stSidebar"] .stFileUploader {
        border: 2px dashed rgba(0, 0, 0, 0.15);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
        background: rgba(0, 0, 0, 0.02);
        margin: 16px 0;
    }
    
    section[data-testid="stSidebar"] .stFileUploader:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.05);
    }
    
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #2c2c2c !important;
        font-size: 14px;
    }
    
    section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }
    
    /* Sidebar columns styling */
    section[data-testid="stSidebar"] [data-testid="column"] {
        padding: 0 4px;
    }
    
    /* Sidebar expander styling */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        color: #2c2c2c;
        padding: 12px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(0, 0, 0, 0.05);
        border-color: rgba(0, 0, 0, 0.12);
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 12px;
    }
    
    /* Sidebar text area styling */
    section[data-testid="stSidebar"] .stTextArea textarea {
        background: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        color: #2c2c2c;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 12px;
    }
    
    section[data-testid="stSidebar"] .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
    }
    
    /* Sidebar spinner styling */
    section[data-testid="stSidebar"] .stSpinner {
        border-color: #3b82f6 !important;
    }
    
    /* Sidebar warning/success/error styling */
    section[data-testid="stSidebar"] .stAlert {
        border-radius: 8px;
        border: none;
        margin: 8px 0;
        font-size: 14px;
    }
    
    section[data-testid="stSidebar"] .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
    }
    
    section[data-testid="stSidebar"] .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #f59e0b;
    }
    
    section[data-testid="stSidebar"] .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }
    
    /* Sidebar section dividers */
    section[data-testid="stSidebar"] hr {
        border: none;
        height: 1px;
        background: rgba(0, 0, 0, 0.08);
        margin: 20px 0;
    }
    
    /* Main content area */
    .main .block-container {
        background: transparent;
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
        margin: 0 auto;
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Chat message styling - similar to Claude's message bubbles */
    .chat-message {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .chat-message:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
    
    /* User message styling - slightly different background */
    .user-message {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.15);
        margin-left: 40px;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.06);
        margin-right: 40px;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(59, 130, 246, 0.08); /* Match user-message */
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 14px 18px;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: #2c2c2c;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.04);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
        outline: none;
    }
    
    /* Button styling - Claude-inspired */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Markdown styling improvements */
    .stMarkdown {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #2c2c2c;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .stMarkdown code {
        background: rgba(0, 0, 0, 0.05);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 0.9em;
    }
    
    .stMarkdown pre {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
        padding: 16px;
        overflow-x: auto;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        color: #6b7280;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.8);
        color: #374151;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1f2937 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
        transition: background 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Loading animation */
    .stSpinner {
        border-color: #3b82f6 !important;
    }
    
    /* Metrics styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    /* Data frame styling */
    .stDataFrame {
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed rgba(0, 0, 0, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.02);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: #3b82f6;
        border-radius: 4px;
    }
    
    .stProgress .st-bn {
        background: rgba(59, 130, 246, 0.1);
        border-radius: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom utility classes */
    .claude-container {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .claude-title {
        font-size: 24px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 16px;
        text-align: center;
    }
    
    .claude-subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
    # Claude-styled sidebar implementation
with st.sidebar:
    linkedin_profile_link = "https://www.linkedin.com/in/abdul-rehman-57a192241/"
    github_profile_link = "https://github.com/AbdulRehman5592/"

    # Social links with better spacing
    st.markdown(
        
        f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <a href="{linkedin_profile_link}" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" 
                     style="margin: 2px; border-radius: 6px;">
            </a>
            <br>
            <a href="{github_profile_link}" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" 
                     style="margin: 2px; border-radius: 6px;">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add a subtle divider
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Model selection with better styling
    st.markdown("### 🤖 Model Selection")
    model_name = st.radio(
        "Choose your AI model:",
        ("Google AI",),
        help="Select the AI model to use for processing"
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Menu section
    st.markdown("### 📋 Menu")
    
    # Action buttons with better layout
    col1, col2 = st.columns(2)
    with col1:
        clear_button = st.button("🔄 Rerun", help="Re-run the last query")
    with col2:
        reset_button = st.button("🗑️ Reset", help="Clear all conversation history")
    
    # Handle button actions
    if reset_button:
        st.session_state.conversation_history = []
        st.session_state.session_id = None
        if st.session_state.session_id:
            requests.post(f"{BACKEND_URL}/reset/", data={"session_id": st.session_state.session_id})
        st.success("✅ Session reset successfully!")
        
    elif clear_button:
        if st.session_state.conversation_history:
            st.warning("⚠️ The previous query will be discarded.")
            st.session_state.conversation_history.pop()
        else:
            st.warning("⚠️ The question in the input will be queried again.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📄 Document Upload")
    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=['pdf'],
        help="Select one or more PDF files to process"
    )
    
    # Process button with better styling
    if st.button("🚀 Submit & Process", help="Process uploaded PDF files"):
        if pdf_docs:
            with st.spinner("🔄 Processing your documents..."):
                try:
                    files = [("files", (pdf.name, pdf, pdf.type)) for pdf in pdf_docs]
                    data = {}
                    if st.session_state.session_id:
                        data["session_id"] = st.session_state.session_id
                    
                    response = requests.post(f"{BACKEND_URL}/upload_pdfs/", files=files, data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.session_id = result["session_id"]
                        st.success(f"✅ Successfully uploaded {len(pdf_docs)} PDF(s). Chunks: {result['chunks']}")
                    else:
                        try:
                            error_msg = response.json().get("error", "Failed to upload PDFs.")
                        except Exception:
                            error_msg = response.text or "Failed to upload PDFs. (Non-JSON response)"
                        st.error(f"❌ {error_msg}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please upload PDF files before processing.")
    
    # Base64 encoding section (only show if PDFs are uploaded)
    if pdf_docs:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🔧 Advanced Tools")
        
        if st.button("📝 Encode PDFs to Base64", help="Convert PDFs to Base64 format"):
            import base64
            base64_results = []
            
            with st.spinner("🔄 Encoding PDFs..."):
                for pdf in pdf_docs:
                    pdf_bytes = pdf.read()
                    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                    base64_results.append({"filename": pdf.name, "base64": b64})
                
                st.success(f"✅ Encoded {len(pdf_docs)} PDF(s) to Base64")
                
                for result in base64_results:
                    with st.expander(f"📄 Base64 for {result['filename']}"):
                        st.text_area(
                            "Base64 String",
                            result["base64"],
                            height=150,
                            help="Copy this Base64 string for external use"
                        )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Workflow diagram button
    st.markdown("### 📊 Visualization")
    if st.button("🔍 Show Workflow Diagram", help="Display the system workflow"):
        st.session_state.show_workflow = True
        st.success("✅ Workflow diagram will be displayed in the main area")
    
    # Add some footer info
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 12px; margin-top: 20px;">
            <p>🤖 Claude-styled Interface</p>
            <p>Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Show the workflow diagram if requested
    if st.session_state.get("show_workflow"):
        st.subheader("Workflow Diagram (Mermaid)")
        mermaid_code = visualize_workflow_mermaid()
        st_mermaid(mermaid_code)
        # Optionally, add a button to hide the diagram
        if st.button("Hide Workflow Diagram"):
            st.session_state.show_workflow = False

    # --- OCR TXT Viewer Section ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📝 OCR Text File Viewer")
    if st.session_state.session_id:
        ocr_dir = f"pdf_output/{st.session_state.session_id}"
        if os.path.exists(ocr_dir):
            txt_files = [f for f in os.listdir(ocr_dir) if f.endswith("_text.txt")]
            if txt_files:
                selected_txt = st.selectbox("Select OCR text file (by page):", txt_files)
                if selected_txt:
                    file_path = os.path.join(ocr_dir, selected_txt)
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    total_lines = len(lines)
                    # Let user pick a line to highlight
                    highlight_line = st.number_input(
                        f"Line number to highlight (1-{total_lines}):", min_value=1, max_value=total_lines, value=1
                    )
                    # Display file with highlight
                    highlighted_text = ""
                    for idx, line in enumerate(lines, 1):
                        if idx == highlight_line:
                            highlighted_text += f'<span style="background-color: #ffe066;"><b>{line.rstrip()}</b></span>\n'
                        else:
                            highlighted_text += line.rstrip() + "\n"
                    st.markdown(f"#### Contents of `{selected_txt}` (highlighted line {highlight_line})", unsafe_allow_html=True)
                    st.markdown(f'<pre style="font-family:monospace;">{highlighted_text}</pre>', unsafe_allow_html=True)
            else:
                st.info("No OCR .txt files found for this session.")
        else:
            st.info("No OCR output directory found for this session.")


if __name__ == "__main__":
    main() 