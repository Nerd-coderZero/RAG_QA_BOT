import streamlit as st
import os
from typing import List
import tempfile
from pathlib import Path
import logging
from HybridDocQA import HybridDocQA  # Update this import to your actual module

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_qa_system():
    """Initialize the QA system with error handling."""
    try:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        cohere_api_key = st.secrets["COHERE_API_KEY"]
        
        # Initialize QA system
        return HybridDocQA(pinecone_api_key, cohere_api_key)
    except Exception as e:
        st.error(f"Error initializing QA system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return None

def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to a temporary directory and return the path."""
    try:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return Path(tmp_file.name)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        st.error(f"Error saving file: {str(e)}")
        return None

def process_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Path]:
    """Process multiple uploaded files and return their paths."""
    file_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                file_paths.append(file_path)
    return file_paths

def display_answer(response: dict):
    """Display the QA response in a formatted way."""
    st.markdown("### Answer")
    st.write(response["answer"])

    if 'sources' in response and response['sources']:
        st.markdown("### Sources")
        for source in response['sources']:
            st.write(f"- {source}")

def main():
    st.set_page_config(page_title="Document QA System", page_icon="📚", layout="wide")
    st.title("📚 Document QA System")
    st.write("Upload documents and ask questions about their content!")

    # Initialize QA system in session state if not already present
    if 'qa_system' not in st.session_state:
        qa_system = initialize_qa_system()
        if qa_system is None:
            st.error("Failed to initialize the QA system. Please check the logs.")
            st.stop()
        st.session_state.qa_system = qa_system

    # File upload section
    with st.form("upload_form"):
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx']
        )
        
        submit_button = st.form_submit_button("Process Documents")
        
        if submit_button and uploaded_files:
            with st.spinner("Processing documents..."):
                file_paths = process_uploaded_files(uploaded_files)
                
                for file_path in file_paths:
                    try:
                        result = st.session_state.qa_system.process_document(file_path)
                        st.write(result)
                        os.unlink(file_path)
                    except Exception as e:
                        st.error(f"Error processing {file_path.name}: {str(e)}")
                        logger.error(f"Document processing error: {str(e)}")
                
                st.success("Documents processed successfully!")

    # Question input section
    question = st.text_input("Ask a question about your documents:", placeholder="Type your question here...")

    if question:
        try:
            with st.spinner("Processing..."):
                response = st.session_state.qa_system.answer_query(question)
                display_answer(response)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            logger.error(f"Question processing error: {str(e)}")

    # Sidebar information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This Document QA System uses advanced AI to answer questions based on uploaded documents.
        
        ### Features
        - Supports PDF, TXT, and DOCX files
        - Provides confidence scores for answers
        - Lists document sources for context
        
        ### How to Use
        1. Upload your documents
        2. Click "Process Documents"
        3. Ask questions about the content
        """)

if __name__ == "__main__":
    main()
