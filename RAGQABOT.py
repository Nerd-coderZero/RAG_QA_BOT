import os
import pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import torch
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import docx2txt
from pdfminer.high_level import extract_text
import traceback
import re
from collections import defaultdict, OrderedDict, Counter
import spacy
import logging
from typing import List, Dict, Any, Union, Optional
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
import json
import tempfile

# Constants for easy tweaking
MAX_CHUNK_SIZE = 1000
BATCH_SIZE = 32
TOP_K = 5  # Top k results to retrieve from Pinecone

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLTKDownloader:
    @staticmethod
    def ensure_nltk_data():
        required_packages = ['punkt', 'stopwords', 'punkt_tab']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
                logger.info(f"NLTK package '{package}' already downloaded")
            except LookupError:
                logger.info(f"Downloading NLTK package '{package}'")
                nltk.download(package, quiet=True)

class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = {'.txt', '.pdf', '.docx'}
        
    def process_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Process an uploaded file from bytes content"""
        try:
            # Create a temporary file
            suffix = Path(filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            # Process the temporary file
            result = self.process_single_file(Path(tmp_path))
            
            # Clean up
            os.unlink(tmp_path)
            return result
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            return f"Error processing uploaded file: {str(e)}"

    def process_document(self, file_path: Union[str, Path]) -> str:
        try:
            path = Path(file_path)
            if path.is_dir():
                return self.process_directory(path)
            if not path.exists():
                return f"Error: File not found - {file_path}"
            return self.process_single_file(path)
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return f"Error processing document: {str(e)}"

    def process_directory(self, directory_path: Path) -> str:
        return "\n".join(self.process_single_file(file) for file in directory_path.glob('*')
                        if file.suffix.lower() in self.supported_extensions)

    def process_single_file(self, file_path: Path) -> str:
        file_extension = file_path.suffix.lower()
        try:
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            elif file_extension == '.pdf':
                return extract_text(str(file_path))
            elif file_extension == '.docx':
                return docx2txt.process(str(file_path))
            else:
                return f"Unsupported file type: {file_extension}"
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return f"Error processing {file_path.name}: {str(e)}"

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_wikipedia(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            # First try exact query
            try:
                page = wikipedia.page(query, auto_suggest=False)
                return {
                    'title': page.title,
                    'content': page.content,
                    'url': page.url
                }
            except DisambiguationError as e:
                # If disambiguation, try the first suggested topic
                if e.options:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    return {
                        'title': page.title, 
                        'content': page.content,
                        'url': page.url
                    }
            except PageError:
                # If no exact match, try search
                search_results = wikipedia.search(query, results=1)
                if search_results:
                    page = wikipedia.page(search_results[0], auto_suggest=False)
                    return {
                        'title': page.title,
                        'content': page.content, 
                        'url': page.url
                    }
        except Exception as e:
            logger.error(f"Error scraping Wikipedia: {str(e)}")
        return None

    def scrape_web(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            return self.scrape_wikipedia(query)
        except Exception as e:
            logger.error(f"Error scraping web: {str(e)}")
            return None

class EnhancedRAGQABot:
    def __init__(self, api_key: str, index_name: str):
        """
        Initialize the RAG QA Bot with the given API key and index name.
        
        Args:
            api_key (str): Pinecone API key
            index_name (str): Name of the Pinecone index to use
        """
        self.initialize_components(api_key, index_name)
        self.web_scraper = WebScraper()
        self.loaded_files = set()
        
    def process_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Process an uploaded file from a web interface.
        
        Args:
            file_content (bytes): The content of the uploaded file
            filename (str): Original filename with extension
            
        Returns:
            str: Status message indicating success or failure
        """
        try:
            content = self.document_processor.process_uploaded_file(file_content, filename)
            if content.startswith("Error:"):
                return content
                
            chunks = self._chunk_text(content)
            self._index_chunks(chunks, source=filename)
            self.loaded_files.add(filename)
            return f"Successfully processed and indexed file: {filename}"
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            return f"Error processing uploaded file: {str(e)}"

    def initialize_components(self, api_key: str, index_name: str):
        try:
            self.pc = pinecone.Pinecone(api_key=api_key)
            if index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            self.index = self.pc.Index(index_name)
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            self.document_processor = DocumentProcessor()
            self._init_spacy()
        except Exception as e:
            logger.error(f"Error initializing RAG QA Bot: {str(e)}")
            raise

    def _init_spacy(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info("Downloading spaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    def dynamic_search(self, query: str) -> bool:
        """Perform dynamic web search and index the results"""
        web_data = self.web_scraper.scrape_web(query)
        if web_data:
            chunks = self._chunk_text(web_data['content'])
            self._index_chunks(chunks, source=web_data['url'])
            return True
        return False

    def load_document(self, file_path: Union[str, Path]) -> str:
        try:
            logger.info(f"Processing document: {file_path}")
            content = self.document_processor.process_document(file_path)
            if content.startswith("Error:"):
                return content
                
            chunks = self._chunk_text(content)
            self._index_chunks(chunks, source=str(file_path))
            self.loaded_files.add(str(Path(file_path)))
            return f"Successfully indexed document: {file_path}"
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            return f"Error loading document: {str(e)}"

    def _chunk_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _index_chunks(self, chunks: List[str], source: Optional[str] = None):
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            try:
                embeddings = self.embedding_model.encode(batch)
                vectors = [{
                    'id': f"{source}_{i + j}" if source else f"chunk_{int(time.time())}_{i + j}",
                    'values': embedding.tolist(),
                    'metadata': {'text': chunk, 'source': source or 'unknown'}
                } for j, (chunk, embedding) in enumerate(zip(batch, embeddings))]
                
                if vectors:
                    self.index.upsert(vectors=vectors)
            except Exception as e:
                logger.error(f"Error indexing batch: {str(e)}")

    def answer_query(self, query: str) -> Dict[str, Any]:
        try:
            local_response = self._get_local_answer(query)
            
            # If confidence is low, try web search
            if local_response['confidence'] < 0.5:
                logger.info("Low confidence, attempting web search...")
                if self.dynamic_search(query):
                    web_response = self._get_local_answer(query)
                    if web_response['confidence'] > local_response['confidence']:
                        return web_response
                        
            return self._generate_fallback_response(local_response)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "I encountered an error while processing your query.",
                "confidence": 0.0,
                "sources": []
            }

    def _get_local_answer(self, query: str) -> Dict[str, Any]:
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        results = self.index.query(
            vector=query_embedding,
            top_k=TOP_K,
            include_metadata=True
        )
        return self._process_results(results['matches'], query)

    def _process_results(self, matches: List[Dict], query: str) -> Dict[str, Any]:
        if not matches:
            return {
                "answer": "No relevant information found.",
                "confidence": 0.0,
                "sources": []
            }
        
        relevant_texts = [match['metadata']['text'] for match in matches]
        try:
            doc = self.nlp(" ".join(relevant_texts))
            query_doc = self.nlp(query)
            scored_sentences = self._score_sentences(doc, query_doc)
            
            if not scored_sentences:
                return {
                    "answer": "No good answer found.",
                    "confidence": 0.1,
                    "sources": []
                }
                
            answer = " ".join([sent for sent, _ in scored_sentences[:3]])
            avg_score = sum(score for _, score in scored_sentences[:3]) / 3
            
            return {
                "answer": answer,
                "confidence": min(avg_score, 1.0),
                "sources": [match['metadata'].get('source', 'unknown') for match in matches[:3]]
            }
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return {
                "answer": "Error processing the results.",
                "confidence": 0.0,
                "sources": []
            }

    def _score_sentences(self, doc, query_doc):
        query_keywords = {token.lemma_.lower() for token in query_doc 
                         if not token.is_stop and not token.is_punct}
        scored_sentences = []
        
        for sent in doc.sents:
            if len(sent.text.split()) < 3:
                continue
                
            sent_keywords = {token.lemma_.lower() for token in sent 
                           if not token.is_stop and not token.is_punct}
            keyword_overlap = len(query_keywords.intersection(sent_keywords))
            similarity_score = sent.similarity(query_doc)
            length_score = min(len(sent.text.split()) / 10, 1.0)
            structure_score = 0.3 if any(token.pos_ == "VERB" for token in sent) and \
                                   any(token.dep_ == "nsubj" for token in sent) else 0
                                   
            final_score = (keyword_overlap * 0.4 + 
                         similarity_score * 0.4 + 
                         length_score * 0.1 + 
                         structure_score * 0.1)
                         
            scored_sentences.append((sent.text, final_score))
            
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return scored_sentences

    def _generate_fallback_response(self, local_response: Dict[str, Any]) -> Dict[str, Any]:
        if local_response['confidence'] < 0.3:
            return {
                "answer": "I couldn't find a confident answer. Here's what I found: " + local_response['answer'],
                "confidence": local_response['confidence'],
                "sources": local_response.get('sources', [])
            }
        return local_response

def main():
    api_key = os.getenv("PINECONE_API_KEY", "your-default-api-key")  # Better to use environment variable
    index_name = "rag-qa"

    try:
        NLTKDownloader.ensure_nltk_data()
        qa_bot = EnhancedRAGQABot(api_key, index_name)

        print("Enhanced RAG QA Bot initialized and ready!")
        print("Commands:")
        print("  load <file_path> - Load and index a document")
        print("  quit - Exit the program")

        while True:
            user_input = input("\nEnter your question or command: ").strip()

            if user_input.lower() == 'quit':
                print("Thank you for using the Enhanced RAG QA Bot. Goodbye!")
                break

            if user_input.lower().startswith('load '):
                file_path = user_input[5:].strip()
                result = qa_bot.load_document(file_path)
                print(result)
            else:
                print("Processing your query...")
                response = qa_bot.answer_query(user_input)

                print(f"\nAnswer: {response['answer']}")
                if 'confidence' in response:
                    print(f"Confidence: {response['confidence']:.2f}")
                if 'sources' in response:
                    sources = [s for s in response['sources'] if s != 'unknown']
                    if sources:
                        print(f"Sources: {', '.join(sources)}")

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
