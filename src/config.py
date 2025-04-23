import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    # General settings
    "output_directory": "output",
    "log_level": "INFO",
    "log_file": os.path.join("logs", "cv_pipeline.log"),
    
    # Chunking settings
    "chunking_strategy": "layout",  # Options: "layout", "heading", "hybrid"
    "extract_bullet_points": True,
    "merge_adjacent_chunks": True,
    "max_chunk_gap": 2,  # Maximum line gap for merging adjacent chunks
    
    # Embedding model settings
    "embedding_model_type": "huggingface_api",  # Options: "sentence_transformer", "huggingface_api", "custom"
    "embedding_model": "BAAI/bge-large-en-v1.5",  # Default model name
    "use_classifier": False,  # Whether to use section classifier
    
    # Vector DB settings
    "vector_store_type": "chroma",  # Options: "chroma", "faiss"
    "vector_store_directory": os.path.join("output", "vector_store"),
    "populate_vector_store": True,  # Whether to add chunks to vector store
    
    # Processing settings
    "save_intermediate_results": False,  # Whether to save intermediate processing results
    
    # Default section headings mapping
    "section_mappings": {
        "experience": "Experience",
        "work experience": "Experience",
        "professional experience": "Experience",
        "employment": "Experience",
        "employment history": "Experience",
        
        "education": "Education",
        "academic": "Education",
        "qualifications": "Education",
        "educational background": "Education",
        
        "skills": "Skills",
        "technical skills": "Skills",
        "core competencies": "Skills",
        "key skills": "Skills",
        "expertise": "Skills",
        "technical expertise": "Skills",
        
        "projects": "Projects",
        "personal projects": "Projects",
        "project experience": "Projects",
        "academic projects": "Projects",
        "research projects": "Projects",
        
        "certifications": "Certifications",
        "certificates": "Certifications",
        "licenses": "Certifications",
        
        "publications": "Publications",
        "papers": "Publications",
        "articles": "Publications",
        "research": "Publications",
        "publications and research": "Publications",
        "journal publications": "Publications",
        
        "languages": "Languages",
        "language proficiency": "Languages",
        "language skills": "Languages",
        "proficiency in": "Languages",
        
        "interests": "Interests",
        "hobbies": "Interests",
        "activities": "Interests",
        
        "summary": "Summary",
        "professional summary": "Summary",
        "profile": "Summary",
        "objective": "Summary",
        "about me": "Summary",
        "career objective": "Summary",
        "about": "Summary",
        
        "achievements": "Achievements",
        "accomplishments": "Achievements",
        "awards": "Achievements",
        
        "volunteer": "Volunteer",
        "volunteer experience": "Volunteer",
        "community service": "Volunteer",
        
        "references": "References",
        "testimonials": "References",
        
        "contact": "Contact",
        "contact information": "Contact",
        "personal information": "Contact",
        "personal details": "Contact",
        "contact details": "Contact",
        "contact information": "Contact",
        "personal information": "Contact"
    }
}

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        import json
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Update default config with user settings
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    # Deep merge dictionaries
                    config[key].update(value)
                else:
                    # Replace value
                    config[key] = value
                    
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading config from {config_path}: {str(e)}")
            print("Using default configuration")
    
    # Load from environment variables
    env_mappings = {
        "EMBEDDING_MODEL_TYPE": "embedding_model_type",
        "EMBEDDING_MODEL": "embedding_model",
        "EMBEDDING_MODEL_BGE": "embedding_model_bge",
        "EMBEDDING_MODEL_E5LARGE": "embedding_model_e5large",
        "VECTOR_STORE_TYPE": "vector_store_type",
        "VECTOR_STORE_DIR": "vector_store_directory",
        "OUTPUT_DIR": "output_directory",
        "LOG_LEVEL": "log_level",
        "DEBUG": "debug"
    }
    
    for env_var, config_key in env_mappings.items():
        if os.getenv(env_var):
            if env_var == "DEBUG":
                # Convert string to boolean
                config[config_key] = os.getenv(env_var).lower() == "true"
            else:
                config[config_key] = os.getenv(env_var)
    
    return config

def get_section_name(heading: str) -> str:
    """
    Get normalized section name from heading
    
    Args:
        heading: Original section heading
        
    Returns:
        Normalized section name
    """
    heading_lower = heading.lower().strip()
    
    # Try direct match
    if heading_lower in DEFAULT_CONFIG["section_mappings"]:
        return DEFAULT_CONFIG["section_mappings"][heading_lower]
    
    # Try partial match
    for key, value in DEFAULT_CONFIG["section_mappings"].items():
        if key in heading_lower:
            return value
    
    return "Other" 