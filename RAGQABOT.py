import cohere
import pinecone
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List, Union
import logging
from pathlib import Path
import docx2txt
from pdfminer.high_level import extract_text

class HybridDocQA:
    def __init__(self, pinecone_api_key: str, cohere_api_key: str = None):
        # Initialize Cohere
        self.co = cohere.Client(cohere_api_key or 'trial')
        
        # Initialize Pinecone for document storage
        self.pc = pinecone.Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index("rag-qa")
        
        # For document embedding
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # Track processed documents
        self.processed_docs = set()

    def process_document(self, file_path: Union[str, Path]) -> str:
        """Process and store document content"""
        try:
            # Extract text based on file type
            path = Path(file_path)
            if path.suffix.lower() == '.pdf':
                text = extract_text(str(path))
            elif path.suffix.lower() == '.docx':
                text = docx2txt.process(str(path))
            elif path.suffix.lower() == '.txt':
                text = path.read_text(encoding='utf-8')
            else:
                return f"Unsupported file type: {path.suffix}"

            # Generate document summary for context
            summary = self.co.summarize(
                text=text,
                length='medium',
                format='paragraph',
                model='command',
                additional_command='Focus on main topics and key information'
            )

            # Store document and summary in Pinecone
            doc_embedding = self.embedder.encode(text)
            summary_embedding = self.embedder.encode(summary.summary)
            
            self.index.upsert([
                {
                    'id': f"doc_{path.name}",
                    'values': doc_embedding.tolist(),
                    'metadata': {
                        'type': 'document',
                        'content': text[:1000],  # Store truncated content
                        'summary': summary.summary,
                        'source': str(path)
                    }
                },
                {
                    'id': f"summary_{path.name}",
                    'values': summary_embedding.tolist(),
                    'metadata': {
                        'type': 'summary',
                        'content': summary.summary,
                        'source': str(path)
                    }
                }
            ])
            
            self.processed_docs.add(str(path))
            return f"Successfully processed and indexed: {path.name}"
            
        except Exception as e:
            return f"Error processing document: {str(e)}"

    def answer_query(self, query: str) -> Dict[str, Any]:
        try:
            # First, check if we have relevant document context
            query_embedding = self.embedder.encode(query)
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=3,
                include_metadata=True
            )

            # Build context from relevant documents
            context = ""
            sources = []
            if results.matches:
                for match in results.matches:
                    if match.score > 0.7:  # Only use high-confidence matches
                        context += f"\nFrom {match.metadata['source']}:\n{match.metadata['content']}\n"
                        sources.append(match.metadata['source'])

            # Generate response using Cohere
            prompt = f"""Task: Answer the question based on the provided context and your knowledge.
            
            Context: {context if context else 'No specific document context available.'}
            
            Question: {query}
            
            Instructions:
            - If the question is about the documents, use the context to answer
            - If no relevant context is found, provide a general answer
            - Be clear about whether the answer comes from documents or general knowledge
            - Keep the answer concise but informative
            
            Answer:"""

            response = self.co.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )

            answer_text = response.generations[0].text.strip()
            
            return {
                "answer": answer_text,
                "sources": sources if sources else ["General knowledge"],
                "has_document_context": bool(context)
            }

        except Exception as e:
            logging.error(f"Error in answer_query: {str(e)}")
            return {
                "answer": "Error processing your query",
                "sources": [],
                "has_document_context": False
            }

# Updated Streamlit interface
def main():
    st.set_page_config(page_title="Document QA System", layout="wide")
    
    if 'qa_system' not in st.session_state:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        cohere_api_key = st.secrets["COHERE_API_KEY"]
        st.session_state.qa_system = HybridDocQA(pinecone_api_key, cohere_api_key)

    st.title("📚 Document Q&A System")
    
    # File upload section
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx']
        )
        
        if uploaded_files:
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                    tmp_file.write(file.getvalue())
                    result = st.session_state.qa_system.process_document(tmp_file.name)
                    st.write(result)
                    os.unlink(tmp_file.name)

    # Question input
    query = st.text_input("Ask a question:", placeholder="Enter your question here...")
    
    if query:
        with st.spinner("Processing..."):
            response = st.session_state.qa_system.answer_query(query)
            
            st.markdown("### Answer")
            st.write(response["answer"])
            
            if response["sources"]:
                st.markdown("### Sources")
                for source in response["sources"]:
                    st.write(f"- {source}")
