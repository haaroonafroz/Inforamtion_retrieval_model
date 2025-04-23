import os
import logging
from typing import List, Dict, Any, Union, Optional
import numpy as np
import torch
import requests
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class EmbeddingModel:
    """Base class for embedding models"""
    
    def __init__(self):
        self.model_name = None
        self.model = None
        self.dimension = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string"""
        raise NotImplementedError("Subclasses must implement this method")


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        logger.debug(f"Embedding {len(texts)} documents")
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string"""
        logger.debug(f"Embedding query: {query[:50]}...")
        embedding = self.model.encode(query, convert_to_tensor=False)
        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding


class HuggingFaceAPIEmbedding(EmbeddingModel):
    """Embedding model using Hugging Face Inference API"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", api_key: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_API_KEY")
        
        if not self.api_key:
            raise ValueError("Hugging Face API key must be provided or set as HF_API_KEY environment variable")
        
        logger.info(f"Initializing Hugging Face API embedding model: {model_name}")
        
        # Set dimension based on the model
        if "bge-large" in model_name:
            self.dimension = 1024
        elif "e5-large" in model_name:
            self.dimension = 1024
        elif "gte-large" in model_name:
            self.dimension = 1024
        else:
            # Default to 768 for most models
            self.dimension = 384
            
        logger.info(f"Hugging Face model initialized with dimension: {self.dimension}")
        
        # API endpoint
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def _get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Get embeddings from Hugging Face Inference API"""
        is_single = isinstance(texts, str)
        payload = {"inputs": texts if is_single else texts}
        
        # Add model-specific parameters
        if "bge" in self.model_name:
            # BGE models work best with specific parameters
            payload["options"] = {"wait_for_model": True}
        elif "e5" in self.model_name:
            # E5 models need query/passage prefix
            if is_single:
                payload["inputs"] = "query: " + texts
            else:
                payload["inputs"] = ["query: " + t for t in texts]
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise exception for error status codes
            result = response.json()
            
            if isinstance(result, list) and all(isinstance(item, list) for item in result):
                return result
            elif isinstance(result, list) and all(isinstance(item, float) for item in result):
                return result
            else:
                raise ValueError(f"Unexpected API response format: {result}")
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        logger.debug(f"Embedding {len(texts)} documents with Hugging Face API")
        
        # Process in batches to avoid API limits
        batch_size = 8  # Smaller batch size for API calls
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_embeddings = self._get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string"""
        logger.debug(f"Embedding query with Hugging Face API: {query[:50]}...")
        return self._get_embeddings(query)


class CustomTransformerEmbedding(EmbeddingModel):
    """Custom implementation of transformer-based embeddings using local models"""
    
    def __init__(self, model_name: str = "intfloat/e5-small-v2"):
        super().__init__()
        self.model_name = model_name
        logger.info(f"Loading custom transformer model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Get embedding dimension based on hidden size
        self.dimension = self.model.config.hidden_size
        logger.info(f"Model loaded with embedding dimension: {self.dimension}")
    
    def _get_embeddings(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Internal method to get embeddings from the model"""
        # Add prefix for some models like E5
        if "e5" in self.model_name:
            if isinstance(text, list):
                text = ["query: " + t if not t.startswith("query: ") else t for t in text]
            else:
                text = "query: " + text if not text.startswith("query: ") else text
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        ).to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling for sentence embeddings
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        
        # Mask padded tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Convert to list and normalize
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if len(embeddings) == 1:
            return embeddings[0].tolist()
        return embeddings.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        logger.debug(f"Embedding {len(texts)} documents with custom transformer")
        
        # Process in batches to avoid CUDA OOM
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            batch_embeddings = self._get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string"""
        logger.debug(f"Embedding query with custom transformer: {query[:50]}...")
        return self._get_embeddings(query)


class SectionClassifier:
    """
    Model to classify text chunks into section types
    (e.g., "Experience", "Education", "Skills")
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the section classifier
        
        Args:
            model_name: The name of the pre-trained model to use
        """
        logger.info(f"Initializing SectionClassifier with model: {model_name}")
        
        try:
            from transformers import AutoModelForSequenceClassification, pipeline
            
            self.model_name = model_name
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create classification pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Define section classes
            self.section_classes = [
                "Experience", "Education", "Skills", 
                "Projects", "Summary", "Contact", 
                "Certifications", "Languages", "Interests",
                "Publications", "References", "Other"
            ]
            
            logger.info("SectionClassifier initialized successfully")
        except ImportError:
            logger.error("Failed to import transformers. Please install with: pip install transformers")
            raise
    
    def predict_section(self, text: str) -> str:
        """
        Predict the section type for a given text
        
        Args:
            text: The text to classify
            
        Returns:
            The predicted section type
        """
        logger.debug(f"Predicting section for text: {text[:50]}...")
        
        # For a proper implementation, this should use a model fine-tuned on resume sections
        # This is a placeholder using text patterns
        text_lower = text.lower()
        
        if any(term in text_lower for term in ["experience", "work", "employment", "career"]):
            return "Experience"
        elif any(term in text_lower for term in ["education", "degree", "university", "college", "school"]):
            return "Education"
        elif any(term in text_lower for term in ["skill", "proficien", "competen"]):
            return "Skills"
        elif any(term in text_lower for term in ["project"]):
            return "Projects"
        elif any(term in text_lower for term in ["summary", "profile", "objective", "about"]):
            return "Summary"
        elif any(term in text_lower for term in ["contact", "phone", "email", "address"]):
            return "Contact"
        elif any(term in text_lower for term in ["certificate", "certification", "license"]):
            return "Certifications"
        elif any(term in text_lower for term in ["language", "fluent", "proficient in"]):
            return "Languages"
        elif any(term in text_lower for term in ["interest", "hobby", "hobbies", "activity"]):
            return "Interests"
        elif any(term in text_lower for term in ["publication", "journal", "conference", "paper"]):
            return "Publications"
        elif any(term in text_lower for term in ["reference"]):
            return "References"
        else:
            return "Other"


# Factory function to create embedding model
def create_embedding_model(model_type: str = "sentence_transformer", **kwargs) -> EmbeddingModel:
    """
    Factory function to create an embedding model
    
    Args:
        model_type: Type of embedding model ("sentence_transformer", "huggingface_api", "custom")
        **kwargs: Additional arguments for the specific model
    
    Returns:
        An instance of EmbeddingModel
    """
    logger.info(f"Creating embedding model of type: {model_type}")
    
    if model_type == "sentence_transformer":
        model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbedding(model_name=model_name)
    
    elif model_type == "huggingface_api":
        model_name = kwargs.get("model_name", "BAAI/bge-large-en-v1.5")
        api_key = kwargs.get("api_key", None)
        return HuggingFaceAPIEmbedding(model_name=model_name, api_key=api_key)
    
    elif model_type == "custom":
        model_name = kwargs.get("model_name", "intfloat/e5-small-v2")
        return CustomTransformerEmbedding(model_name=model_name)
    
    else:
        error_msg = f"Unknown model type: {model_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)


# Example usage
if __name__ == "__main__":
    # Test sentence transformer embedding
    model = create_embedding_model("sentence_transformer")
    test_texts = ["This is a test document", "This is another document"]
    embeddings = model.embed_documents(test_texts)
    print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
    
    # Test section classifier
    classifier = SectionClassifier()
    section = classifier.predict_section("Work Experience")
    print(f"Predicted section: {section}")