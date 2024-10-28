import streamlit as st
import os
from typing import List
import tempfile
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import RAG components with error handling
try:
    from ragqabot import EnhancedRAGQABot, NLTKDownloader
except ImportError as e:
    st.error(f"Failed to import RAG components: {str(e)}")
    st.stop()

def initialize_qa_bot():
    """Initialize the QA bot with proper error handling"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["PINECONE_API_KEY"]
        index_name = "rag-qa"
        
        # Ensure NLTK data is downloaded
        NLTKDownloader.ensure_nltk_data()
        
        # Initialize QA bot
        return EnhancedRAGQABot(api_key, index_name)
    except Exception as e:
        st.error(f"Error initializing QA bot: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return None

def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to temporary directory and return the path"""
    try:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return Path(tmp_file.name)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        st.error(f"Error saving file: {str(e)}")
        return None

def process_uploaded_files(uploaded_files: List[st.uploaded_file_manager.UploadedFile]) -> List[Path]:
    """Process multiple uploaded files and return their paths"""
    file_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                file_paths.append(file_path)
    return file_paths

def display_answer(response: dict):
    """Display the QA response in a formatted way"""
    st.markdown("### Answer")
    st.write(response["answer"])

    st.markdown("### Confidence Score")
    confidence = float(response['confidence'])
    st.progress(confidence)
    st.write(f"{confidence:.2f}")

    if 'sources' in response and response['sources']:
        st.markdown("### Sources")
        sources = [s for s in response['sources'] if s != 'unknown']
        if sources:
            for source in sources:
                st.write(f"- {source}")

def main():
    st.set_page_config(
        page_title="RAG QA Bot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("📚 RAG QA Bot")
    st.write("Upload documents and ask questions about their content!")

    # Initialize QA bot in session state if not already present
    if 'qa_bot' not in st.session_state:
        qa_bot = initialize_qa_bot()
        if qa_bot is None:
            st.error("Failed to initialize QA bot. Please check the logs.")
            st.stop()
        st.session_state.qa_bot = qa_bot

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )

    if uploaded_files:
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    file_paths = process_uploaded_files(uploaded_files)
                    
                    for file_path in file_paths:
                        try:
                            result = st.session_state.qa_bot.load_document(file_path)
                            st.write(result)
                            # Clean up temporary file
                            os.unlink(file_path)
                        except Exception as e:
                            st.error(f"Error processing {file_path.name}: {str(e)}")
                            logger.error(f"Document processing error: {str(e)}")
                    
                    st.success("Documents processed successfully!")

    # Question input section
    question = st.text_input("Ask a question about your documents:", 
                           placeholder="Type your question here...")

    if question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_bot.answer_query(question)
                display_answer(response)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            logger.error(f"Question processing error: {str(e)}")

    # Sidebar information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This RAG QA Bot uses advanced AI to answer questions about your documents. 
        
        ### Features
        - Process PDF, TXT, and DOCX files
        - Perform web searches for additional context
        - Provide confidence scores for answers
        - List sources for verification
        
        ### How to Use
        1. Upload your documents
        2. Click "Process Documents"
        3. Ask questions about the content
        """)

if __name__ == "__main__":
    main()
