import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGChain

# LangChain imports
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGSystem:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, use_pinecone: bool = False):
        """
        Initialize Enhanced RAG System

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between adjacent chunks
            use_pinecone: Whether to use Pinecone instead of FAISS
        """
        # Validate API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        if use_pinecone and not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_pinecone = use_pinecone

        # Initialize components
        self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Setup Pinecone config if needed
        pinecone_config = None
        if use_pinecone:
            pinecone_config = {
                'api_key': os.getenv("PINECONE_API_KEY"),
                'environment': os.getenv("PINECONE_ENVIRONMENT")
            }

        self.vector_store_manager = VectorStoreManager(
            use_pinecone=use_pinecone,
            pinecone_config=pinecone_config
        )
        self.rag_chain = None
        self.chunks = None

        logging.info(f"RAG System initialized with chunk_size={chunk_size}, overlap={chunk_overlap}, pinecone={use_pinecone}")

    def load_and_process_pdf(self, pdf_path: str, analyze: bool = True) -> List[Document]:
        """
        Load and process PDF using enhanced PDF processor

        Args:
            pdf_path: Path to PDF file
            analyze: Whether to analyze the document structure

        Returns:
            List of document chunks
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logging.info(f"Loading PDF: {pdf_path}")

        # Process PDF with fallback approach
        chunks = self.pdf_processor.process_pdf_complete(pdf_path)

        if analyze:
            # Analyze original documents for insights
            original_docs = self.pdf_processor.load_pdf_with_fallback(pdf_path)
            analysis = self.pdf_processor.analyze_document_structure(original_docs)
            self._display_analysis(analysis)

        self.chunks = chunks
        logging.info(f"Successfully processed PDF into {len(chunks)} chunks")
        return chunks

    def _display_analysis(self, analysis: dict):
        """Display document analysis results"""
        print(f"\n{'='*50}")
        print("DOCUMENT ANALYSIS")
        print(f"{'='*50}")
        print(f"Total documents: {analysis['total_documents']}")
        print(f"Total characters: {analysis['total_characters']:,}")
        print(f"Total words: {analysis['total_words']:,}")
        print(f"Unique pages: {analysis['unique_pages']}")
        print(f"Loader types: {analysis['loader_types']}")

        if 'avg_document_size' in analysis:
            print(f"Average document size: {analysis['avg_document_size']:.0f} characters")
            print(f"Size range: {analysis['min_document_size']} - {analysis['max_document_size']} characters")

    def create_vector_store(self, chunks: List[Document], save_path: str = "faiss_index", index_name: str = "financial-investing-rag"):
        """
        Create and save vector store

        Args:
            chunks: Document chunks
            save_path: Path to save FAISS index (ignored for Pinecone)
            index_name: Index name for Pinecone
        """
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")

        logging.info(f"Creating vector embeddings for {len(chunks)} chunks...")

        # Create vector store
        if self.use_pinecone:
            self.vector_store_manager.create_vector_store(chunks, index_name=index_name)
            logging.info(f"Vector store created in Pinecone index: {index_name}")
        else:
            self.vector_store_manager.create_vector_store(chunks)
            self.vector_store_manager.save_local(save_path)
            logging.info(f"FAISS vector store saved to {save_path}")

    def load_vector_store(self, load_path: str = "faiss_index"):
        """Load existing FAISS vector store"""
        if self.use_pinecone:
            logging.warning("Cannot load from path when using Pinecone. Vector store should be already initialized.")
            return

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Vector store not found: {load_path}")

        self.vector_store_manager.load_local(load_path)
        logging.info(f"Vector store loaded from {load_path}")

    def setup_qa_chain(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.2):
        """
        Setup the question-answering chain

        Args:
            model_name: OpenAI model to use
            temperature: Temperature for response generation (lower = more focused)
        """
        if not self.vector_store_manager.vector_store:
            raise ValueError("Vector store not initialized")

        self.rag_chain = RAGChain(
            self.vector_store_manager,
            model_name=model_name,
            temperature=temperature
        )
        logging.info("QA chain setup complete")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            question: User question

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.rag_chain:
            raise ValueError("QA chain not setup. Call setup_qa_chain() first.")

        logging.info(f"Processing query: {question[:50]}...")

        # Process the question
        try:
            result = self.rag_chain.query(question)

            # Add additional metadata
            result.update({
                "question": question,
                "num_sources": len(result.get("source_documents", [])),
                "cost_usd": result.get("cost", 0.0)
            })

            logging.info(f"Query completed. Tokens: {result.get('tokens_used', 0)}, Cost: ${result.get('cost_usd', 0):.4f}")
            return result

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            raise

    def get_relevant_chunks(self, question: str, k: int = 5) -> List[Dict]:
        """Get relevant chunks for inspection"""
        if not self.vector_store_manager.vector_store:
            raise ValueError("Vector store not initialized")

        docs = self.vector_store_manager.similarity_search(question, k=k)

        chunk_info = []
        for i, doc in enumerate(docs):
            chunk_info.append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "length": len(doc.page_content),
                "page": doc.metadata.get('page', 'Unknown')
            })

        return chunk_info

    def setup_from_pdf(self, pdf_path: str, analyze: bool = True) -> List[Document]:
        """Complete setup from PDF file"""
        # Process PDF
        chunks = self.load_and_process_pdf(pdf_path, analyze=analyze)

        # Create vector store
        index_name = f"rag-{Path(pdf_path).stem}" if self.use_pinecone else None
        self.create_vector_store(chunks, index_name=index_name)

        # Setup QA chain
        self.setup_qa_chain()

        logging.info("RAG system fully initialized!")
        return chunks

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "use_pinecone": self.use_pinecone,
            "total_chunks": len(self.chunks) if self.chunks else 0,
            "vector_store_type": "Pinecone" if self.use_pinecone else "FAISS",
            "model": "gpt-3.5-turbo",
            "embeddings": "text-embedding-ada-002"
        }
        return stats

def main():
    """Main function to run the enhanced RAG system"""
    print("-------- Financial Education PDF RAG System --------")
    print("=" * 55)

    PDF_PATH = "data/nfer-doc.pdf"
    VECTOR_STORE_PATH = "data/faiss_index"
    USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() == "true"

    try:
        # Initialize enhanced RAG system
        rag = RAGSystem(
            chunk_size=1000,
            chunk_overlap=200,
            use_pinecone=USE_PINECONE
        )

        # Check if PDF exists
        if not Path(PDF_PATH).exists():
            print(f"❌ PDF not found: {PDF_PATH}")
            print("Please ensure your PDF is in the correct location.")
            return

        # Setup based on vector store type
        if USE_PINECONE:
            print("🌲 Using Pinecone vector database...")
            chunks = rag.setup_from_pdf(PDF_PATH, analyze=True)
            print(f"✅ Setup complete with Pinecone! Created {len(chunks)} chunks.")
        else:
            # Check if FAISS vector store exists
            if Path(VECTOR_STORE_PATH).exists():
                print("📁 Found existing FAISS vector store. Loading...")
                rag.load_vector_store(VECTOR_STORE_PATH)
                rag.setup_qa_chain()
                print("✅ FAISS vector store loaded successfully!")
            else:
                print("📄 Processing PDF and creating FAISS vector store...")
                chunks = rag.setup_from_pdf(PDF_PATH, analyze=True)
                print(f"✅ Setup complete with FAISS! Created {len(chunks)} chunks.")

        # Interactive session
        print("\n🤖 Enhanced RAG System Ready!")
        print("Commands:")
        print("  • Ask any question about the document")
        print("  • Type 'chunks <question>' to see relevant chunks")
        print("  • Type 'stats' to see system statistics")
        print("  • Type 'help' for more commands")
        print("  • Type 'quit' to exit")

        last_question = ""

        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break

                elif user_input.lower() == 'help':
                    print("\n📖 Available Commands:")
                    print("  • Ask questions about the document content")
                    print("  • 'chunks <question>' - Show relevant document chunks")
                    print("  • 'stats' - Show system statistics")
                    print("  • 'last' - Repeat last question")
                    print("  • 'quit' - Exit the system")
                    continue

                elif user_input.lower() == 'stats':
                    stats = rag.get_system_stats()
                    print("\n📊 System Statistics:")
                    for key, value in stats.items():
                        print(f"  • {key.replace('_', ' ').title()}: {value}")
                    continue

                elif user_input.lower() == 'last' and last_question:
                    user_input = last_question

                elif user_input.lower().startswith('chunks '):
                    question = user_input[7:].strip()
                    if question:
                        print(f"\n🔍 Relevant chunks for: '{question}'")
                        print("-" * 50)

                        relevant_chunks = rag.get_relevant_chunks(question, k=3)

                        for chunk in relevant_chunks:
                            print(f"\n📄 Chunk {chunk['rank']} (Page {chunk['page']}):")
                            print(f"   Length: {chunk['length']} characters")
                            print(f"   Content: {chunk['content'][:200]}...")
                    continue

                # Process regular question
                print("🔍 Searching for answer...")
                result = rag.query(user_input)
                last_question = user_input

                # Display results
                print(f"\n✨ Answer:")
                print("-" * 30)
                print(result['answer'])

                print(f"\n📋 Query Info:")
                print(f"  • Sources used: {result['num_sources']} chunks")
                print(f"  • Tokens used: {result.get('tokens_used', 0)}")
                print(f"  • Cost: ${result['cost_usd']:.4f}")

                # Show source preview
                if result.get('source_documents'):
                    print(f"\n📚 Source Preview:")
                    for i, doc in enumerate(result['source_documents'][:2]):
                        page = doc.metadata.get('page', 'Unknown')
                        print(f"  Source {i+1} (Page {page}): {doc.page_content[:100]}...")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logging.error(f"Error in main loop: {e}")

    except Exception as e:
        print(f"❌ System initialization error: {e}")
        logging.error(f"System initialization error: {e}")

if __name__ == "__main__":
    main()