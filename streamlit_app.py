import streamlit as st
import os
from typing import List
import tempfile
from io import BytesIO
import sys
import subprocess

# First, install required packages if not present
def install_required_packages():
    required = {
        'spacy': 'spacy',
        'torch': 'torch',
        'numpy': 'numpy',
        'sentence_transformers': 'sentence-transformers',
        'pinecone-client': 'pinecone-client',
        'nltk': 'nltk',
        'wikipedia': 'wikipedia',
        'python-docx': 'python-docx',
        'pdfminer.six': 'pdfminer.six'
    }
    
    for package, version in required.items():
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", version])

    # Install spacy model
    try:
        import spacy
        spacy.load('en_core_web_sm')
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Install required packages
install_required_packages()

# Now import the main RAG bot code
from ragqabot import EnhancedRAGQABot, NLTKDownloader

def initialize_session_state():
    """Initialize the session state variables"""
    if 'qa_bot' not in st.session_state:
        try:
            # Initialize the QA bot with Pinecone API key
            api_key = st.secrets["PINECONE_API_KEY"]
            index_name = "rag-qa"
            
            # Ensure NLTK data is downloaded
            NLTKDownloader.ensure_nltk_data()
            
            # Initialize the QA bot
            st.session_state.qa_bot = EnhancedRAGQABot(api_key, index_name)
            st.session_state.processed_files = set()
            st.success("✅ QA Bot initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing QA bot: {str(e)}")
            st.stop()

def process_file(uploaded_file) -> bool:
    """Process a single uploaded file"""
    try:
        # Get file content
        file_content = uploaded_file.read()
        
        # Process the file content
        result = st.session_state.qa_bot.process_uploaded_file(file_content, uploaded_file.name)
        
        if not result.startswith("Error"):
            st.session_state.processed_files.add(uploaded_file.name)
            return True
        else:
            st.error(result)
            return False
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="RAG QA Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🤖 RAG QA Bot")
    st.markdown("Upload documents and ask questions about their content!")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, TXT, DOCX)",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )
    
    # Process uploaded files
    if uploaded_files:
        if st.button("Process Documents"):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            processed_count = 0
            for idx, file in enumerate(uploaded_files):
                progress_text.text(f"Processing {file.name}...")
                if process_file(file):
                    processed_count += 1
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            progress_text.empty()
            progress_bar.empty()
            
            if processed_count > 0:
                st.success(f"✅ Successfully processed {processed_count} documents!")
    
    # Question input section
    question = st.text_input("💭 Ask a question about your documents:")
    
    if question:
        if not st.session_state.processed_files:
            st.warning("⚠️ Please upload and process some documents first!")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_bot.answer_query(question)
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.write(response["answer"])
                    
                    # Display confidence score
                    confidence = float(response["confidence"])
                    st.markdown("### 📊 Confidence Score")
                    st.progress(confidence)
                    st.write(f"{confidence:.2f}")
                    
                    # Display sources
                    if 'sources' in response and response['sources']:
                        st.markdown("### 📚 Sources")
                        sources = [s for s in response['sources'] if s != 'unknown']
                        if sources:
                            for source in sources:
                                st.write(f"- {source}")
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
    
    # Sidebar
    st.sidebar.header("📋 Processed Files")
    if st.session_state.get('processed_files'):
        for file in st.session_state.processed_files:
            st.sidebar.success(f"✅ {file}")
    else:
        st.sidebar.info("No files processed yet")

if __name__ == "__main__":
    main()
