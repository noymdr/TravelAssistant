"""
RAG System for Travel Documents
Retrieves information from local documents first, then enriches with LLM knowledge if needed.
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_ollama import Ollama as OllamaLLM

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


class TravelRAGSystem:
    """RAG system that prioritizes source documents and enriches with LLM knowledge when needed."""
    
    def __init__(
        self,
        documents_dir: str = "documents",
        vector_store_path: str = "vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "gemma2"
    ):
        """
        Initialize the RAG system.
        
        Args:
            documents_dir: Directory containing travel documents
            vector_store_path: Path to save/load FAISS vector store
            embedding_model: Hugging Face embedding model name
            ollama_base_url: URL for Ollama API
            llm_model: Ollama model name (e.g., "gemma2")
        """
        self.documents_dir = Path(documents_dir)
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True)
        
        # Initialize embeddings
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM
        print("Initializing LLM...")
        try:
            # Try new API style first
            self.llm = OllamaLLM(
                base_url=ollama_base_url,
                model=llm_model,
                temperature=0.1
            )
        except TypeError:
            # Fallback to older API style
            self.llm = OllamaLLM(
                model=llm_model,
                temperature=0.1
            )
        
        # Initialize vector store
        self.vector_store = None
        self.documents = []
        self.embeddings_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Load or create vector store
        self._load_or_create_vector_store()
    
    def _load_documents(self) -> List[Document]:
        """Load all documents from the documents directory."""
        documents = []
        
        if not self.documents_dir.exists():
            self.documents_dir.mkdir(exist_ok=True)
            print(f"Created documents directory: {self.documents_dir}")
            return documents
        
        # Supported file types
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        txt_files = list(self.documents_dir.glob("*.txt"))
        
        print(f"Found {len(pdf_files)} PDF files and {len(txt_files)} text files")
        
        # Load PDF files
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                # Add metadata about source file
                for doc in docs:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {pdf_file.name}")
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
        
        # Load text files
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source_file'] = txt_file.name
                    doc.metadata['file_type'] = 'txt'
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {txt_file.name}")
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
        
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def _create_vector_store(self, documents: List[Document]):
        """Create FAISS vector store from documents."""
        if not documents:
            print("No documents to index. Please add documents to the documents directory.")
            return
        
        print("Splitting documents into chunks...")
        split_docs = self._split_documents(documents)
        self.documents = split_docs
        
        print(f"Creating vector store with {len(split_docs)} chunks...")
        texts = [doc.page_content for doc in split_docs]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Get embedding dimension
        self.embeddings_dim = embeddings_array.shape[1]
        
        # Create FAISS index
        self.vector_store = faiss.IndexFlatL2(self.embeddings_dim)
        self.vector_store.add(embeddings_array)
        
        # Save vector store and documents
        self._save_vector_store()
        print(f"Vector store created with {len(split_docs)} documents")
    
    def _save_vector_store(self):
        """Save vector store and documents metadata."""
        if self.vector_store is not None:
            # Save FAISS index
            faiss.write_index(self.vector_store, str(self.vector_store_path / "index.faiss"))
            
            # Save document metadata
            import pickle
            with open(self.vector_store_path / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            print(f"Vector store saved to {self.vector_store_path}")
    
    def _load_vector_store(self) -> bool:
        """Load existing vector store."""
        index_path = self.vector_store_path / "index.faiss"
        documents_path = self.vector_store_path / "documents.pkl"
        
        if not index_path.exists() or not documents_path.exists():
            return False
        
        try:
            # Load FAISS index
            self.vector_store = faiss.read_index(str(index_path))
            self.embeddings_dim = self.vector_store.d
            print(f"Loaded vector store with dimension {self.embeddings_dim}")
            
            # Load documents
            import pickle
            with open(documents_path, "rb") as f:
                self.documents = pickle.load(f)
            
            print(f"Loaded {len(self.documents)} documents from vector store")
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one."""
        if not self._load_vector_store():
            documents = self._load_documents()
            if documents:
                self._create_vector_store(documents)
    
    def _retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents from vector store."""
        if self.vector_store is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS
        k = min(k, len(self.documents))
        distances, indices = self.vector_store.search(query_vector, k)
        
        # Retrieve documents
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs
    
    def _check_if_answer_in_sources(self, query: str, retrieved_docs: List[Document]) -> bool:
        """Check if the answer can be found in the retrieved documents."""
        if not retrieved_docs:
            return False
        
        # Combine retrieved document content
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Use LLM to check if answer is in sources
        check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Based on the following context from travel documents, can you answer this question?

Question: {query}

Context from documents:
{context}

Answer with only "YES" if the question can be answered from the context, or "NO" if the information is not available in the context.
"""
        )
        
        chain = check_prompt | self.llm
        response = chain.invoke({"query": query, "context": context})
        # Convert to string if needed (handles AIMessage objects)
        response_str = str(response) if not isinstance(response, str) else response
        
        return "YES" in response_str.upper()
    
    def query(self, question: str, use_enrichment: bool = True, k: int = 5) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            use_enrichment: Whether to enrich with LLM knowledge if not in sources
            k: Number of documents to retrieve
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        print(f"\nQuery: {question}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self._retrieve_documents(question, k=k)
        
        if not retrieved_docs:
            if use_enrichment:
                print("No documents found. Using LLM knowledge base...")
                answer = self.llm.invoke(question)
                # Convert to string if needed (handles AIMessage objects)
                answer = str(answer) if not isinstance(answer, str) else answer
                return {
                    "answer": answer,
                    "sources": [],
                    "source_based": False,
                    "enriched": True
                }
            else:
                return {
                    "answer": "No relevant information found in the provided documents.",
                    "sources": [],
                    "source_based": False,
                    "enriched": False
                }
        
        # Step 2: Check if answer is in sources
        answer_in_sources = self._check_if_answer_in_sources(question, retrieved_docs)
        
        # Step 3: Generate answer
        context = "\n\n".join([
            f"[From {doc.metadata.get('source_file', 'unknown')}]:\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        
        if answer_in_sources:
            # Answer from sources only
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful travel assistant. Answer the question based ONLY on the information provided in the context from the travel documents. Do not use any external knowledge.

Context from travel documents:
{context}

Question: {question}

Answer based only on the information in the context:
"""
            )
        else:
            # Enrich with LLM knowledge
            if use_enrichment:
                prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""You are a helpful travel assistant. Answer the question using information from the travel documents when available, and supplement with your knowledge when needed.

Context from travel documents:
{context}

Question: {question}

Answer the question. Use information from the documents when available, and supplement with your knowledge when the information is not in the documents:
"""
                )
            else:
                prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""You are a helpful travel assistant. Answer the question based ONLY on the information provided in the context from the travel documents.

Context from travel documents:
{context}

Question: {question}

Answer based only on the information in the context:
"""
                )
        
        chain = prompt_template | self.llm
        answer = chain.invoke({"context": context, "question": question})
        # Convert to string if needed (handles AIMessage objects)
        answer = str(answer) if not isinstance(answer, str) else answer
        
        # Prepare sources
        sources = [
            {
                "file": doc.metadata.get('source_file', 'unknown'),
                "file_type": doc.metadata.get('file_type', 'unknown'),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for doc in retrieved_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "source_based": answer_in_sources,
            "enriched": not answer_in_sources and use_enrichment
        }
    
    def rebuild_index(self):
        """Rebuild the vector store from documents."""
        documents = self._load_documents()
        if documents:
            self._create_vector_store(documents)
        else:
            print("No documents found to index.")


def main():
    """Example usage of the RAG system."""
    # Initialize the system
    rag = TravelRAGSystem(
        documents_dir="documents",
        vector_store_path="vector_store",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ollama_base_url="http://localhost:11434",
        llm_model="gemma2"
    )
    
    # Example queries
    queries = [
        "When is my flight?",
        "What's the hotel phone number?",
        "What are the reviews about the hotel?",
    ]
    
    for query in queries:
        result = rag.query(query)
        print(f"\n{'='*60}")
        print(f"Question: {query}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSource-based: {result['source_based']}")
        print(f"Enriched: {result['enriched']}")
        if result['sources']:
            print(f"\nSources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['file']} ({source['file_type']})")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

