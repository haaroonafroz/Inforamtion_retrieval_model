import os
import uuid
import logging
import shutil
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from src.pipeline import CVProcessingPipeline
from src.config import load_config
from src.utils import save_json, load_json

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load configuration
config = load_config()

# Create upload directory
UPLOAD_DIR = os.path.join("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize API
app = FastAPI(
    title="CV-StructRAG API",
    description="API for structured information retrieval from CV/Resume PDFs",
    version="1.0.0"
)

# Initialize pipeline
pipeline = CVProcessingPipeline(config=config)

# Keep track of processed documents
document_store = {}

# Response models
class ProcessingStatus(BaseModel):
    document_id: str
    status: str
    filename: str

class StructuredData(BaseModel):
    document_id: str
    filename: str
    data: Dict[str, Any]

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    section_filter: Optional[str] = None
    results: List[SearchResult]

class HealthStatus(BaseModel):
    status: str
    version: str

# Process PDF in background
def process_pdf_task(file_path: str, document_id: str):
    try:
        # Process the PDF
        result = pipeline.process_pdf(file_path, save_output=True)
        
        # Store the result
        document_store[document_id] = {
            "filename": os.path.basename(file_path),
            "status": "completed",
            "data": result["structured_data"]
        }
        
        logger.info(f"Successfully processed document {document_id}")
    except Exception as e:
        # Update status to error
        document_store[document_id] = {
            "filename": os.path.basename(file_path),
            "status": "error",
            "error": str(e)
        }
        logger.error(f"Error processing document {document_id}: {str(e)}")

@app.post("/upload-pdf", response_model=ProcessingStatus)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a CV/Resume PDF file for processing
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Create directory for this document
    document_dir = os.path.join(UPLOAD_DIR, document_id)
    os.makedirs(document_dir, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(document_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Store initial status
    document_store[document_id] = {
        "filename": file.filename,
        "status": "processing",
        "data": None
    }
    
    # Process in background
    background_tasks.add_task(process_pdf_task, file_path, document_id)
    
    return {
        "document_id": document_id,
        "status": "processing",
        "filename": file.filename
    }

@app.get("/processing-status/{document_id}", response_model=ProcessingStatus)
async def get_processing_status(document_id: str):
    """
    Get the processing status of a document
    """
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": document_id,
        "status": document_store[document_id]["status"],
        "filename": document_store[document_id]["filename"]
    }

@app.get("/extract-json/{document_id}", response_model=StructuredData)
async def get_structured_data(document_id: str):
    """
    Get the structured JSON data extracted from a processed document
    """
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = document_store[document_id]
    
    if doc_info["status"] == "processing":
        raise HTTPException(status_code=202, detail="Document is still being processed")
    
    if doc_info["status"] == "error":
        raise HTTPException(status_code=500, detail=f"Error processing document: {doc_info.get('error', 'Unknown error')}")
    
    return {
        "document_id": document_id,
        "filename": doc_info["filename"],
        "data": doc_info["data"]
    }

@app.get("/search", response_model=SearchResponse)
async def search_sections(
    query: str = Query(..., description="Search query"),
    section_filter: Optional[str] = Query(None, description="Optional section filter (e.g., 'Experience', 'Skills')"),
    k: int = Query(5, description="Number of results to return", ge=1, le=20)
):
    """
    Search for relevant sections across all processed documents
    """
    try:
        results = pipeline.search_sections(query, section_filter, k)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result["text"],
                "score": result["score"],
                "metadata": result["metadata"]
            })
        
        return {
            "query": query,
            "section_filter": section_filter,
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its data
    """
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from document store
    document_store.pop(document_id)
    
    # Remove document directory
    document_dir = os.path.join(UPLOAD_DIR, document_id)
    if os.path.exists(document_dir):
        shutil.rmtree(document_dir)
    
    return {"message": f"Document {document_id} deleted successfully"}

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "ok",
        "version": "1.0.0"
    }

# Run the API server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 