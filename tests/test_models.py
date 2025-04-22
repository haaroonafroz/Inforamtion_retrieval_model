import pytest
import numpy as np
from src.models import (
    create_embedding_model, 
    SentenceTransformerEmbedding,
    CustomTransformerEmbedding,
    SectionClassifier
)

class TestEmbeddingModels:
    """Test case for embedding models"""
    
    def test_sentence_transformer_embedding(self):
        """Test SentenceTransformer embedding model"""
        # Skip test if sentence_transformers not installed
        pytest.importorskip("sentence_transformers")
        
        model = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
        
        # Test embedding a single text
        query = "This is a test query"
        embedding = model.embed_query(query)
        
        # Check output shape and type
        assert isinstance(embedding, list)
        assert len(embedding) == model.dimension
        assert all(isinstance(x, float) for x in embedding)
        
        # Test embedding multiple texts
        texts = ["First document", "Second document"]
        embeddings = model.embed_documents(texts)
        
        # Check output
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert len(embeddings[0]) == model.dimension
    
    def test_create_embedding_model(self):
        """Test factory function for creating embedding models"""
        # Skip test if sentence_transformers not installed
        pytest.importorskip("sentence_transformers")
        
        # Test default model
        model = create_embedding_model()
        assert isinstance(model, SentenceTransformerEmbedding)
        assert model.model_name == "all-MiniLM-L6-v2"
        
        # Test with custom model name
        model = create_embedding_model(model_type="sentence_transformer", model_name="all-mpnet-base-v2")
        assert model.model_name == "all-mpnet-base-v2"
        
        # Test with invalid model type
        with pytest.raises(ValueError):
            create_embedding_model(model_type="invalid_type")

class TestSectionClassifier:
    """Test case for section classifier"""
    
    def test_section_classification(self):
        """Test section classification"""
        classifier = SectionClassifier()
        
        # Test various section types
        assert classifier.predict_section("Work Experience") == "Experience"
        assert classifier.predict_section("Senior Software Engineer at Google") == "Experience"
        assert classifier.predict_section("Education and Qualifications") == "Education"
        assert classifier.predict_section("Technical Skills and Competencies") == "Skills"
        assert classifier.predict_section("Python, JavaScript, TypeScript") == "Skills"
        assert classifier.predict_section("Personal Projects") == "Projects"
        assert classifier.predict_section("Professional Summary") == "Summary"
        assert classifier.predict_section("Contact Information") == "Contact"
        assert classifier.predict_section("Certifications and Licenses") == "Certifications"
        assert classifier.predict_section("Language Proficiency") == "Languages"
        assert classifier.predict_section("Hobbies and Interests") == "Interests"
        assert classifier.predict_section("Publications and Research") == "Publications"
        assert classifier.predict_section("References") == "References"
        assert classifier.predict_section("Random text that doesn't fit any category") == "Other" 