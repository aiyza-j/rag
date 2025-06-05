from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Optional
import logging

try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    logging.warning("Pinecone not available. Only FAISS will be supported.")
    PINECONE_AVAILABLE = False

class VectorStoreManager:
    def __init__(self, use_pinecone: bool = False, pinecone_config: dict = None):
        self.embeddings = OpenAIEmbeddings()
        self.use_pinecone = use_pinecone and PINECONE_AVAILABLE
        self.vector_store = None
        self.pinecone_client = None

        if self.use_pinecone and pinecone_config:
            self._init_pinecone(pinecone_config)
        elif use_pinecone and not PINECONE_AVAILABLE:
            logging.warning("Pinecone requested but not available. Falling back to FAISS.")
            self.use_pinecone = False

    def _init_pinecone(self, config: dict):
        """Initialize Pinecone with new API"""
        try:
            self.pinecone_client = Pinecone(api_key=config['api_key'])
            logging.info("Pinecone client initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Pinecone: {e}")
            logging.warning("Falling back to FAISS")
            self.use_pinecone = False
            raise

    def create_vector_store(self, chunks: List[Document], index_name: str = None) -> None:
        """Create vector store from chunks"""
        try:
            if self.use_pinecone and self.pinecone_client:
                # Create or get index
                if index_name not in [index.name for index in self.pinecone_client.list_indexes()]:
                    self.pinecone_client.create_index(
                        name=index_name,
                        dimension=1536,  # OpenAI embedding dimension
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                    logging.info(f"Created new Pinecone index: {index_name}")

                # Create vector store
                self.vector_store = PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    index_name=index_name
                )
                logging.info(f"Created Pinecone vector store with {len(chunks)} chunks")
            else:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                logging.info(f"Created FAISS vector store with {len(chunks)} chunks")
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
            # Fallback to FAISS if Pinecone fails
            if self.use_pinecone:
                logging.warning("Pinecone failed, falling back to FAISS")
                self.use_pinecone = False
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                logging.info(f"Created FAISS vector store with {len(chunks)} chunks (fallback)")
            else:
                raise

    def save_local(self, path: str = "faiss_index"):
        """Save FAISS index locally"""
        if not self.use_pinecone and self.vector_store:
            try:
                self.vector_store.save_local(path)
                logging.info(f"FAISS index saved to {path}")
            except Exception as e:
                logging.error(f"Error saving FAISS index: {e}")
                raise

    def load_local(self, path: str = "faiss_index"):
        """Load FAISS index from local storage"""
        if not self.use_pinecone:
            try:
                self.vector_store = FAISS.load_local(
                    path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logging.info(f"FAISS index loaded from {path}")
            except Exception as e:
                logging.error(f"Error loading FAISS index: {e}")
                raise

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logging.error(f"Error performing similarity search: {e}")
            raise

    def get_retriever(self, search_kwargs: dict = None):
        """Get retriever for RAG chain"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        search_kwargs = search_kwargs or {"k": 5}
        try:
            return self.vector_store.as_retriever(search_kwargs=search_kwargs)
        except Exception as e:
            logging.error(f"Error creating retriever: {e}")
            raise