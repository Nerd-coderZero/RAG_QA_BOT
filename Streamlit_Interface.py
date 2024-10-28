import streamlit as st
import os
from typing import List
import tempfile
from ragqabot import EnhancedRAGQABot, NLTKDownloader

# Initialize session state
if 'qa_bot' not in st.session_state:
    # Initialize the QA bot with your Pinecone API key
    api_key = st.secrets["PINECONE_API_KEY"]  # Store this in Streamlit secrets
    index_name = "rag-qa"
    
    # Ensure NLTK data is downloaded
    NLTKDownloader.ensure_nltk_data()
    
    # Initialize the QA bot
    st.session_state.qa_bot = EnhancedRAGQABot(api_key, index_name)

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return the path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_uploaded_files(uploaded_files: List[st.uploaded_file_manager.UploadedFile]) -> List[str]:
    """Process multiple uploaded files and return their paths"""
    file_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                file_paths.append(file_path)
    return file_paths

def main():
    st.title("RAG QA Bot")
    st.write("Upload documents and ask questions about their content!")

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, TXT, DOCX)", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                file_paths = process_uploaded_files(uploaded_files)
                for file_path in file_paths:
                    result = st.session_state.qa_bot.load_document(file_path)
                    st.write(result)
                    # Clean up temporary file
                    os.unlink(file_path)
            st.success("Documents processed successfully!")

    # Question input section
    question = st.text_input("Ask a question about your documents:")
    
    if question:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_bot.answer_query(question)
            
            # Display the answer in a nice format
            st.write("### Answer")
            st.write(response["answer"])
            
            # Display confidence score with a progress bar
            st.write("### Confidence Score")
            st.progress(float(response["confidence"]))
            st.write(f"{response['confidence']:.2f}")
            
            # Display sources if available
            if 'sources' in response and response['sources']:
                st.write("### Sources")
                sources = [s for s in response['sources'] if s != 'unknown']
                if sources:
                    for source in sources:
                        st.write(f"- {source}")

    # Add some helpful information in the sidebar
    st.sidebar.header("About")
    st.sidebar.write("""
    This RAG QA Bot uses advanced AI to answer questions about your documents.
    It can:
    - Process PDF, TXT, and DOCX files
    - Perform web searches for additional context
    - Provide confidence scores for answers
    - List sources for verification
    """)

if __name__ == "__main__":
    main()
