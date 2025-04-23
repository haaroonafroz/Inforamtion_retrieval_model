# utils.py
import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import pdfplumber

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    # Configure root logger
    handlers = []
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handlers.append(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    handlers.append(console_handler)
    
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers
    )
    
    logger.info(f"Logging configured with level: {log_level}")
# -------------------------------------------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Replace Unicode characters
    text = text.replace('\u2019', "'")
    text = text.replace('\u2018', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u00a0', ' ')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove headers/footers (simple heuristic)
    lines = text.split('\n')
    if len(lines) > 4:
        # Remove potential header (first line if short)
        if len(lines[0]) < 50:
            lines = lines[1:]
        # Remove potential footer (last line if short)
        if len(lines[-1]) < 50:
            lines = lines[:-1]
    
    text = '\n'.join(lines)
    
    return text.strip()

# -------------------------------------------------------------------------------------------------------------

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save with pretty formatting
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    logger.debug(f"Data saved to {filepath}")

# -------------------------------------------------------------------------------------------------------------

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to load from
        
    Returns:
        Loaded data
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# -------------------------------------------------------------------------------------------------------------

def extract_contact_info(text: str) -> Dict[str, str]:
    """
    Extract contact information from text
    
    Args:
        text: Text to extract from
        
    Returns:
        Dictionary of contact information
    """
    contact_info = {}
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, text)
    if email_matches:
        contact_info["email"] = email_matches[0]
    
    # Extract phone
    phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    phone_matches = re.findall(phone_pattern, text)
    if phone_matches:
        contact_info["phone"] = phone_matches[0]
    
    # Extract LinkedIn
    linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=|linkedin\.com/pub/)[A-Za-z0-9_-]+'
    linkedin_matches = re.findall(linkedin_pattern, text.lower())
    if linkedin_matches:
        contact_info["linkedin"] = linkedin_matches[0]
    
    # Extract location (simple city, state pattern)
    location_pattern = r'\b[A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*(?:[\s,]+[A-Z]{2})?\b'
    location_matches = re.findall(location_pattern, text)
    if location_matches:
        # Try to find something that looks like a location
        for match in location_matches:
            if len(match.split()) >= 2 and not re.search(r'@|\.com|www\.', match):
                contact_info["location"] = match
                break
    
    return contact_info

# -------------------------------------------------------------------------------------------------------------

def extract_text_info(pdf_path):
    elements = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            for char in page.chars:
                elements.append({
                    "text": char["text"],
                    "font_size": char["size"],
                    "x0": char["x0"],
                    "x1": char["x1"],
                    "top": char["top"],
                    "bottom": char["bottom"],
                    "page": page_number
                })
    return elements

# -------------------------------------------------------------------------------------------------------------


