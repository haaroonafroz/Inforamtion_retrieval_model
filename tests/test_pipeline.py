import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from src.pipeline import CVProcessingPipeline
from src.utils import save_json

# Sample output data for mocking
SAMPLE_STRUCTURED_DATA = {
    "Experience": {
        "Software Engineer - ABC Corp": [
            "Built APIs",
            "Developed UI"
        ]
    },
    "Skills": [
        "Python", "JavaScript", "React"
    ]
}

class TestCVProcessingPipeline:
    """Test case for CV processing pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "output_directory": self.temp_dir,
            "populate_vector_store": False  # Disable vector store to avoid external dependencies
        }
    
    def teardown_method(self):
        """Tear down test fixtures"""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.chunking.ResumeChunker')
    @patch('src.models.create_embedding_model')
    @patch('src.vectordb.create_vector_store')
    def test_pipeline_initialization(self, mock_create_vector_store, mock_create_embedding_model, mock_chunker):
        """Test pipeline initialization"""
        # Create mocks
        mock_create_embedding_model.return_value = MagicMock()
        mock_create_vector_store.return_value = MagicMock()
        
        # Initialize pipeline
        pipeline = CVProcessingPipeline(config=self.config)
        
        # Check if components are initialized
        assert pipeline.chunker is not None
        assert pipeline.embedding_model is not None
        assert pipeline.vector_store is not None
        
        # Check if output directory is created
        assert os.path.exists(self.temp_dir)
    
    @patch('src.chunking.ResumeChunker')
    def test_process_pdf(self, mock_chunker_class):
        """Test PDF processing"""
        # Setup mock
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Mock process_pdf result
        mock_chunker.process_pdf.return_value = {
            "hierarchical_chunks": [{"heading": "Experience"}, {"heading": "Skills"}],
            "flat_chunks": [{"text": "chunk1"}, {"text": "chunk2"}],
            "output_json": SAMPLE_STRUCTURED_DATA
        }
        
        # Initialize pipeline with mocks
        with patch('src.models.create_embedding_model') as mock_create_embedding_model:
            with patch('src.vectordb.create_vector_store') as mock_create_vector_store:
                mock_create_embedding_model.return_value = MagicMock()
                mock_create_vector_store.return_value = MagicMock()
                
                pipeline = CVProcessingPipeline(config=self.config)
                
                # Process PDF
                result = pipeline.process_pdf("dummy.pdf", save_output=True)
                
                # Check if chunker was called
                mock_chunker.process_pdf.assert_called_once_with("dummy.pdf")
                
                # Check return value
                assert "hierarchical_chunks" in result
                assert "flat_chunks" in result
                assert "structured_data" in result
                assert result["structured_data"] == SAMPLE_STRUCTURED_DATA
                
                # Check if output file was created
                output_file = os.path.join(self.temp_dir, "dummy_processed.json")
                assert os.path.exists(output_file)
    
    @patch('src.vectordb.VectorStore')
    def test_search_sections(self, mock_vector_store):
        """Test section search functionality"""
        # Create mock vector store
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"text": "Python, JavaScript", "metadata": {"normalized_heading": "Skills"}, "score": 0.95},
            {"text": "React, Node.js", "metadata": {"normalized_heading": "Skills"}, "score": 0.85}
        ]
        
        # Initialize pipeline with mock
        with patch('src.chunking.ResumeChunker'):
            with patch('src.models.create_embedding_model'):
                with patch('src.vectordb.create_vector_store', return_value=mock_store):
                    pipeline = CVProcessingPipeline(config=self.config)
                    
                    # Test with no filter
                    results = pipeline.search_sections("Python skills", None, k=5)
                    mock_store.search.assert_called_with(query="Python skills", k=5, filter_metadata=None)
                    assert len(results) == 2
                    
                    # Test with section filter
                    results = pipeline.search_sections("Python skills", "Skills", k=3)
                    filter_metadata = {"normalized_heading": "Skills"}
                    mock_store.search.assert_called_with(query="Python skills", k=3, filter_metadata=filter_metadata)
    
    @patch('src.chunking.ResumeChunker')
    def test_extract_structured_data(self, mock_chunker_class):
        """Test structured data extraction"""
        # Setup mock
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Mock process_pdf result
        mock_chunker.process_pdf.return_value = {
            "output_json": SAMPLE_STRUCTURED_DATA
        }
        
        # Initialize pipeline with mocks
        with patch('src.models.create_embedding_model'):
            with patch('src.vectordb.create_vector_store'):
                pipeline = CVProcessingPipeline(config=self.config)
                
                # Extract structured data
                result = pipeline.extract_structured_data("dummy.pdf")
                
                # Check result
                assert result == SAMPLE_STRUCTURED_DATA
                mock_chunker.process_pdf.assert_called_once_with("dummy.pdf") 