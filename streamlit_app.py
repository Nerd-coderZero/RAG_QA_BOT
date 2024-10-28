import streamlit as st
import os
from typing import List, Optional
import tempfile
from RAGQABOT import EnhancedRAGQABot, NLTKDownloader

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
        except Exception as e:
            st.error(f"Error initializing QA bot: {str(e)}")
            st.stop()

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Save uploaded file to temporary directory and return the path
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Optional[str]: Path to saved file or None if error occurs
    """
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file '{uploaded_file.name}': {str(e)}")
        return None

def process_uploaded_files(uploaded_files: List[st.uploaded_file_manager.UploadedFile]) -> None:
    """
    Process multiple uploaded files and update the session state
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
    """
    progress_bar = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name not in st.session_state.processed_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        result = st.session_state.qa_bot.load_document(file_path)
                        if not result.startswith("Error"):
                            st.session_state.processed_files.add(uploaded_file.name)
                        st.write(result)
                        # Clean up temporary file
                        os.unlink(file_path)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        progress_bar.progress((idx + 1) / len(uploaded_files))
    progress_bar.empty()

def display_answer(response: dict) -> None:
    """
    Display the QA bot's response in a formatted way
    
    Args:
        response: Dictionary containing answer, confidence, and sources
    """
    # Display answer in a card-like container
    with st.container():
        st.markdown("### 💭 Answer")
        st.markdown(f">{response['answer']}")
        
        # Display confidence score with colored progress bar
        st.markdown("### 📊 Confidence Score")
        confidence = float(response['confidence'])
        color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red'
        st.markdown(
            f"""
            <div style="border-radius:20px;padding:10px;background-color:{color};width:{confidence*100}%">
                <p style="color:white;margin:0;text-align:center">{confidence:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display sources if available
        if 'sources' in response and response['sources']:
            st.markdown("### 📚 Sources")
            sources = [s for s in response['sources'] if s != 'unknown']
            if sources:
                for source in sources:
                    st.markdown(f"- {source}")

def create_sidebar() -> None:
    """Create and populate the sidebar"""
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    ### Features
    - 📄 Supports PDF, TXT, and DOCX files
    - 🌐 Dynamic web search capability
    - 📊 Confidence scoring
    - 📚 Source attribution
    
    ### How to use
    1. Upload your documents
    2. Wait for processing to complete
    3. Ask questions about the content
    4. View answers with confidence scores
    
    ### Processing Status
    """)
    if st.session_state.get('processed_files'):
        st.sidebar.markdown("**Processed Files:**")
        for file in st.session_state.processed_files:
            st.sidebar.markdown(f"- ✅ {file}")
    else:
        st.sidebar.markdown("*No files processed yet*")

def main():
    st.set_page_config(
        page_title="RAG QA Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🤖 RAG QA Bot")
    st.markdown("Upload documents and ask questions about their content!")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx']
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                process_uploaded_files(uploaded_files)
                st.success("✅ Documents processed successfully!")
        
        # Question input section
        question = st.text_input("💭 Ask a question about your documents:")
        
        if question:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_bot.answer_query(question)
                    display_answer(response)
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
    
    with col2:
        create_sidebar()

if __name__ == "__main__":
    main()
