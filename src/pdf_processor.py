from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Optional
import logging
from pathlib import Path

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Enhanced PDF processor with reliable PyPDF support

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between adjacent chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf_pypdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF using PyPDFLoader (primary method)

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of documents
        """
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} pages using PyPDFLoader")

            # Add metadata and clean text
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'loader_type': 'pypdf',
                    'original_length': len(doc.page_content),
                    'page_number': i + 1
                })
                # Clean text
                doc.page_content = self._clean_text(doc.page_content)

            # Filter out empty documents
            documents = [doc for doc in documents if len(doc.page_content.strip()) > 50]
            logging.info(f"Filtered to {len(documents)} non-empty documents")

            return documents

        except Exception as e:
            logging.error(f"Error loading PDF with PyPDFLoader: {e}")
            raise

    def load_pdf_with_fallback(self, pdf_path: str) -> List[Document]:
        """
        Load PDF with reliable fallback approach

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of documents
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        documents = []

        try:
            logging.info("Loading PDF with PyPDFLoader...")
            documents = self.load_pdf_pypdf(pdf_path)

            # Check if we got meaningful content
            if not documents:
                raise ValueError("No content extracted from PDF")

            total_content = sum(len(doc.page_content.strip()) for doc in documents)
            if total_content < 100:
                raise ValueError("Insufficient content extracted from PDF")

        except Exception as e:
            logging.error(f"Failed to load PDF: {e}")
            raise

        return documents

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove common PDF artifacts
        artifacts = [
            '\x00', '\x01', '\x02', '\x03',  # Control characters
            '\f',  # Form feed
        ]

        for artifact in artifacts:
            text = text.replace(artifact, '')

        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove multiple consecutive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks with enhanced metadata

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        try:
            chunks = self.text_splitter.split_documents(documents)

            # Add enhanced metadata to chunks
            for i, chunk in enumerate(chunks):
                original_metadata = chunk.metadata.copy()
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content),
                    'source_document': original_metadata.get('source', 'unknown'),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.page_content.split()),
                    'char_count': len(chunk.page_content)
                })

            logging.info(f"Created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logging.error(f"Error splitting documents: {e}")
            raise

    def process_pdf_complete(self, pdf_path: str) -> List[Document]:
        """
        Complete PDF processing pipeline

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of processed document chunks
        """
        # Load documents
        documents = self.load_pdf_with_fallback(pdf_path)

        # Split into chunks
        chunks = self.split_documents(documents)

        # Log processing summary
        self._log_processing_summary(documents, chunks, pdf_path)

        return chunks

    def _log_processing_summary(self, documents: List[Document], chunks: List[Document], pdf_path: str):
        """Log processing summary"""
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0

        loader_types = set(doc.metadata.get('loader_type', 'unknown') for doc in documents)

        logging.info(f"""
        PDF Processing Summary for {Path(pdf_path).name}:
        - Original documents: {len(documents)}
        - Total characters: {total_chars:,}
        - Generated chunks: {len(chunks)}
        - Average chunk size: {avg_chunk_size:.0f} characters
        - Loader types used: {', '.join(loader_types)}
        - Configured chunk size: {self.chunk_size}
        - Configured overlap: {self.chunk_overlap}
                """)

    def analyze_document_structure(self, documents: List[Document]) -> dict:
        """
        Analyze the structure of loaded documents

        Args:
            documents: List of documents to analyze

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_documents': len(documents),
            'total_characters': sum(len(doc.page_content) for doc in documents),
            'total_words': sum(len(doc.page_content.split()) for doc in documents),
            'loader_types': {},
            'document_sizes': [],
            'pages': set(),
            'metadata_keys': set()
        }

        for doc in documents:
            # Collect sizes
            analysis['document_sizes'].append(len(doc.page_content))

            # Collect loader types
            loader_type = doc.metadata.get('loader_type', 'unknown')
            analysis['loader_types'][loader_type] = analysis['loader_types'].get(loader_type, 0) + 1

            # Collect pages
            if 'page' in doc.metadata:
                analysis['pages'].add(doc.metadata['page'])

            # Collect metadata keys
            analysis['metadata_keys'].update(doc.metadata.keys())

        # Calculate statistics
        if analysis['document_sizes']:
            analysis['avg_document_size'] = sum(analysis['document_sizes']) / len(analysis['document_sizes'])
            analysis['min_document_size'] = min(analysis['document_sizes'])
            analysis['max_document_size'] = max(analysis['document_sizes'])

        analysis['unique_pages'] = len(analysis['pages'])
        analysis['metadata_keys'] = list(analysis['metadata_keys'])

        return analysis