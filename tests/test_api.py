import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
from src.api import app

client = TestClient(app)

class TestAPI:
    """Test case for the FastAPI endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create a temporary PDF file for testing
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self.temp_file.write(b"Mock PDF content")
        self.temp_file.close()
    
    def teardown_method(self):
        """Tear down test fixtures"""
        # Remove temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
    
    @patch('src.api.process_pdf_task')
    def test_upload_pdf(self, mock_process_task):
        """Test PDF upload endpoint"""
        # Use the temp file for testing
        with open(self.temp_file.name, "rb") as f:
            response = client.post(
                "/upload-pdf",
                files={"file": ("test.pdf", f, "application/pdf")}
            )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "processing"
        assert data["filename"] == "test.pdf"
        
        # Check if background task was added
        assert mock_process_task.call_count == 1
    
    def test_upload_non_pdf(self):
        """Test upload with non-PDF file"""
        # Create a non-PDF file
        temp_txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        temp_txt.write(b"This is not a PDF")
        temp_txt.close()
        
        try:
            with open(temp_txt.name, "rb") as f:
                response = client.post(
                    "/upload-pdf",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            # Check response
            assert response.status_code == 400
            data = response.json()
            assert "Only PDF files are accepted" in data["detail"]
        finally:
            os.unlink(temp_txt.name)
    
    def test_get_processing_status(self):
        """Test getting processing status"""
        # Mock document in store
        doc_id = "test-doc-id"
        app.document_store[doc_id] = {
            "filename": "test.pdf",
            "status": "completed",
            "data": {}
        }
        
        # Get status
        response = client.get(f"/processing-status/{doc_id}")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc_id
        assert data["status"] == "completed"
        
        # Test non-existent document
        response = client.get("/processing-status/non-existent")
        assert response.status_code == 404
    
    def test_get_structured_data(self):
        """Test getting structured data"""
        # Mock document in store
        doc_id = "test-doc-id"
        test_data = {
            "Experience": {"Job Title": ["Bullet 1", "Bullet 2"]},
            "Skills": ["Skill 1", "Skill 2"]
        }
        
        app.document_store[doc_id] = {
            "filename": "test.pdf",
            "status": "completed",
            "data": test_data
        }
        
        # Get data
        response = client.get(f"/extract-json/{doc_id}")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc_id
        assert data["data"] == test_data
        
        # Test processing document
        processing_id = "processing-doc"
        app.document_store[processing_id] = {
            "filename": "processing.pdf",
            "status": "processing",
            "data": None
        }
        
        response = client.get(f"/extract-json/{processing_id}")
        assert response.status_code == 202
        
        # Test error document
        error_id = "error-doc"
        app.document_store[error_id] = {
            "filename": "error.pdf",
            "status": "error",
            "error": "Test error"
        }
        
        response = client.get(f"/extract-json/{error_id}")
        assert response.status_code == 500
    
    @patch('src.pipeline.CVProcessingPipeline.search_sections')
    def test_search(self, mock_search):
        """Test search endpoint"""
        # Mock search results
        mock_search.return_value = [
            {"text": "Result 1", "metadata": {"heading": "Skills"}, "score": 0.95},
            {"text": "Result 2", "metadata": {"heading": "Experience"}, "score": 0.85}
        ]
        
        # Search
        response = client.get("/search?query=python&section_filter=Skills&k=3")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "python"
        assert data["section_filter"] == "Skills"
        assert len(data["results"]) == 2
        
        # Check if search was called with correct parameters
        mock_search.assert_called_with("python", "Skills", 3)
        
        # Test search with error
        mock_search.side_effect = Exception("Test error")
        response = client.get("/search?query=python")
        assert response.status_code == 500
    
    def test_delete_document(self):
        """Test document deletion"""
        # Mock document in store
        doc_id = "test-doc-id"
        app.document_store[doc_id] = {
            "filename": "test.pdf",
            "status": "completed",
            "data": {}
        }
        
        # Create mock document directory
        doc_dir = os.path.join(app.UPLOAD_DIR, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Delete document
        response = client.delete(f"/documents/{doc_id}")
        
        # Check response
        assert response.status_code == 200
        
        # Check if document was removed from store
        assert doc_id not in app.document_store
        
        # Check if directory was removed
        assert not os.path.exists(doc_dir)
        
        # Test non-existent document
        response = client.delete("/documents/non-existent")
        assert response.status_code == 404 