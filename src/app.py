from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
import logging
from pathlib import Path
import asyncio
import uuid
from datetime import datetime

from main import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Assistant API",
    description="A powerful document Q&A system using RAG (Retrieval Augmented Generation)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    max_sources: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    question: str
    tokens_used: int
    cost: float
    processing_time: float
    timestamp: str

class SystemStats(BaseModel):
    status: str
    chunk_count: int
    vector_store_type: str
    model_name: str
    embeddings_model: str
    is_ready: bool
    pdf_loaded: Optional[str] = None

class ChunkInfo(BaseModel):
    rank: int
    content: str
    page: str
    length: int
    preview: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    processing_time: float
    file_id: str

rag_system: Optional[RAGSystem] = None
current_pdf_path: Optional[str] = None
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    try:
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem(
            chunk_size=1000,
            chunk_overlap=200,
            use_pinecone=os.getenv("USE_PINECONE", "false").lower() == "true"
        )

        # Check if there's an existing vector store to load
        default_pdf = "data/nfer-doc.pdf"
        vector_store_path = "data/faiss_index"

        if Path(default_pdf).exists():
            if Path(vector_store_path).exists() and not rag_system.use_pinecone:
                logger.info("Loading existing vector store...")
                rag_system.load_vector_store(vector_store_path)
                rag_system.setup_qa_chain()
                global current_pdf_path
                current_pdf_path = default_pdf
            else:
                logger.info("Processing default PDF...")
                await process_pdf_background(default_pdf)

        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")

async def process_pdf_background(pdf_path: str):
    """Process PDF in background"""
    global rag_system, current_pdf_path
    try:
        chunks = rag_system.setup_from_pdf(pdf_path, analyze=False)
        current_pdf_path = pdf_path
        logger.info(f"Background processing complete: {len(chunks)} chunks created")
        return len(chunks)
    except Exception as e:
        logger.error(f"Background PDF processing failed: {e}")
        raise

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Document Assistant API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global rag_system, current_pdf_path

    is_ready = (
        rag_system is not None and
        rag_system.rag_chain is not None and
        rag_system.vector_store_manager.vector_store is not None
    )

    return {
        "status": "healthy" if is_ready else "initializing",
        "rag_system_ready": is_ready,
        "pdf_loaded": Path(current_pdf_path).name if current_pdf_path else None,
        "vector_store_type": "Pinecone" if (rag_system and rag_system.use_pinecone) else "FAISS",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload")
):
    """Upload and process a PDF file"""
    global rag_system, current_pdf_path

    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = upload_dir / filename

    try:
        start_time = datetime.now()

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        chunks = await asyncio.to_thread(rag_system.setup_from_pdf, str(file_path), False)
        current_pdf_path = str(file_path)

        processing_time = (datetime.now() - start_time).total_seconds()

        return UploadResponse(
            message="PDF uploaded and processed successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            processing_time=processing_time,
            file_id=file_id
        )

    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            os.remove(file_path)
        logger.error(f"PDF upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document using RAG"""
    global rag_system

    if not rag_system or not rag_system.rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not ready. Please upload a document first.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        start_time = datetime.now()

        # Process query
        result = await asyncio.to_thread(rag_system.query, request.question)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Format sources
        sources = []
        if result.get("source_documents"):
            for i, doc in enumerate(result["source_documents"][:request.max_sources]):
                sources.append({
                    "id": i + 1,
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "page": doc.metadata.get("page", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", i),
                    "length": len(doc.page_content)
                })

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            question=request.question,
            tokens_used=result.get("tokens_used", 0),
            cost=result.get("cost", 0.0),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/chunks/{question}")
async def get_relevant_chunks(question: str, k: int = 5):
    """Get relevant document chunks for a question"""
    global rag_system

    if not rag_system or not rag_system.vector_store_manager.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not ready")

    try:
        chunks = await asyncio.to_thread(rag_system.get_relevant_chunks, question, k)

        formatted_chunks = []
        for chunk in chunks:
            formatted_chunks.append(ChunkInfo(
                rank=chunk["rank"],
                content=chunk["content"],
                page=str(chunk["page"]),
                length=chunk["length"],
                preview=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
            ))

        return {
            "question": question,
            "chunks": formatted_chunks,
            "total_found": len(formatted_chunks)
        }

    except Exception as e:
        logger.error(f"Chunk retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )