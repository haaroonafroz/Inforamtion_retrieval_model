# utils.py
import os
import re
import json
import pdfplumber
import numpy as np
import faiss
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from src.models import *
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

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

def find_sentence_boundary(text: str, position: int, forward: bool = True) -> int:
    """
    Find the nearest sentence boundary from a given position.
    
    Args:
        text: The text to search in
        position: Starting position
        forward: If True, search forward; if False, search backward
    
    Returns:
        Position of the nearest sentence boundary
    """
    endings = {'.', '!', '?', ':', ';'}
    if forward:
        for i in range(position, len(text)):
            if text[i] in endings and (i + 1 == len(text) or text[i + 1].isspace()):
                return i + 1
        return len(text)
    else:
        for i in range(position - 1, -1, -1):
            if i > 0 and text[i - 1] in endings and text[i].isspace():
                return i
        return 0

def create_chunks(text: str, chunk_size: int = 1000, overlap: int = 50, min_chunk_size: int = 100) -> list:
    """
    Create chunks with proper word and sentence preservation.
    
    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for any chunk
    
    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    current_pos = 0
    text_length = len(text)

    while current_pos < text_length:
        chunk_end = min(current_pos + chunk_size, text_length)
        if chunk_end < text_length:
            chunk_end = find_sentence_boundary(text, chunk_end, forward=True)
        chunk = text[current_pos:chunk_end].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        elif chunks:
            chunks[-1] += " " + chunk
        if chunk_end < text_length:
            overlap_start = max(current_pos, chunk_end - overlap)
            current_pos = find_sentence_boundary(text, overlap_start, forward=False)
        else:
            current_pos = chunk_end

    validated_chunks = []
    for chunk in chunks:
        if not chunk[0].isupper() and validated_chunks:
            validated_chunks[-1] += " " + chunk
            continue
        chunk = re.sub(r'\s+', ' ', chunk)
        validated_chunks.append(chunk)

    return validated_chunks

def process_pdf_for_rag(pdf_path: str, output_file: str = None, chunk_size: int = 1000, overlap: int = 50, min_chunk_size: int = 100):
    """
    Process PDF and create chunks suitable for RAG.
    
    Args:
        pdf_path: Path to PDF file
        output_file: Optional path to save chunks
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for any chunk
    """
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += " " + clean_text(text)

    chunks = create_chunks(all_text, chunk_size=chunk_size, overlap=overlap, min_chunk_size=min_chunk_size)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Chunks saved to: {output_file}")

    return chunks

def ensure_folders_exist():
    os.makedirs("embeddings_index", exist_ok=True)
    if not os.path.exists("chat_history.json"):
        with open("chat_history.json", "w") as f:
            json.dump([], f)

def load_chat_history() -> list:
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as f:
            return json.load(f)
    return []

def save_chat_history(chat_history: list):
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)

def clear_chat_history():
    with open("chat_history.json", "w") as f:
        json.dump([], f)

def detect_section_title(line: str) -> bool:
    """
    Detect if a line is likely a section title
    
    Args:
        line: Text line to check
        
    Returns:
        True if the line is likely a section title
    """
    # Common section headers in resumes
    section_patterns = [
        r'(?i)^(professional\s+experience|experience|work\s+experience|employment(\s+history)?)$',
        r'(?i)^(education(\s+and\s+training)?)$',
        r'(?i)^(skills|technical\s+skills|core\s+competencies)$',
        r'(?i)^(projects|personal\s+projects)$',
        r'(?i)^(certifications|certificates)$',
        r'(?i)^(publications)$',
        r'(?i)^(languages)$',
        r'(?i)^(interests|hobbies)$',
        r'(?i)^(summary|profile|professional\s+summary|about\s+me)$',
        r'(?i)^(accomplishments|achievements)$',
        r'(?i)^(volunteer(\s+experience)?)$',
        r'(?i)^(references)$',
        r'(?i)^(contact(\s+information)?)$',
    ]
    
    # Check if line matches any pattern
    for pattern in section_patterns:
        if re.search(pattern, line.strip()):
            return True
    
    # Heuristics for section titles
    line = line.strip()
    if line and line == line.upper() and len(line) < 30:
        return True
    
    if line and line.strip().endswith(':') and len(line) < 30:
        return True
    
    return False

def extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from text
    
    Args:
        text: Text containing bullet points
        
    Returns:
        List of bullet point texts
    """
    bullet_patterns = [
        r'•\s+([^\n•]+)',
        r'■\s+([^\n■]+)',
        r'○\s+([^\n○]+)',
        r'▪\s+([^\n▪]+)',
        r'✓\s+([^\n✓]+)',
        r'✔\s+([^\n✔]+)',
        r'-\s+([^\n-]+)',
        r'\*\s+([^\n\*]+)',
        r'\d+\.\s+([^\n]+)'
    ]
    
    bullet_points = []
    remaining_text = text
    
    for pattern in bullet_patterns:
        matches = re.findall(pattern, remaining_text)
        if matches:
            bullet_points.extend([match.strip() for match in matches])
            # Remove matched content for next pattern
            remaining_text = re.sub(pattern, '', remaining_text)
    
    # If no bullet points found but there are line breaks, treat lines as points
    if not bullet_points and '\n' in text:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Filter out very short lines (likely not content)
        bullet_points = [line for line in lines if len(line) > 10]
    
    return bullet_points

def merge_adjacent_chunks(chunks: List[Dict[str, Any]], max_gap: int = 2) -> List[Dict[str, Any]]:
    """
    Merge adjacent chunks with the same section type if they're within max_gap lines
    
    Args:
        chunks: List of chunks
        max_gap: Maximum number of lines between chunks to merge
        
    Returns:
        List of merged chunks
    """
    if not chunks or len(chunks) < 2:
        return chunks
    
    merged_chunks = [chunks[0]]
    
    for i in range(1, len(chunks)):
        current = chunks[i]
        previous = merged_chunks[-1]
        
        # Check if chunks are from the same section and within max_gap
        same_section = (
            current.get('metadata', {}).get('normalized_heading') == 
            previous.get('metadata', {}).get('normalized_heading')
        )
        
        prev_line_num = previous.get('metadata', {}).get('line_number', 0)
        curr_line_num = current.get('metadata', {}).get('line_number', 0)
        close_enough = (curr_line_num - prev_line_num) <= max_gap
        
        if same_section and close_enough:
            # Merge content
            previous['content'] = previous['content'] + '\n' + current['content']
            
            # Merge subsections if they exist
            if 'subsections' in previous and 'subsections' in current:
                previous['subsections'].extend(current['subsections'])
        else:
            merged_chunks.append(current)
    
    return merged_chunks

def normalize_section_name(section_name: str) -> str:
    """
    Normalize section name to a standard name
    
    Args:
        section_name: Original section name
        
    Returns:
        Normalized section name
    """
    section_name = section_name.lower().strip()
    
    if re.search(r'experience|employment|work', section_name):
        return "Experience"
    elif re.search(r'education|academic|qualification', section_name):
        return "Education"
    elif re.search(r'skill|competenc|proficienc', section_name):
        return "Skills"
    elif re.search(r'project', section_name):
        return "Projects"
    elif re.search(r'certification|certificate', section_name):
        return "Certifications"
    elif re.search(r'publication', section_name):
        return "Publications"
    elif re.search(r'language', section_name):
        return "Languages"
    elif re.search(r'interest|hobby|hobbies', section_name):
        return "Interests"
    elif re.search(r'summary|profile|objective|about', section_name):
        return "Summary"
    elif re.search(r'achievement|accomplishment', section_name):
        return "Achievements"
    elif re.search(r'volunteer', section_name):
        return "Volunteer"
    elif re.search(r'reference', section_name):
        return "References"
    elif re.search(r'contact|address|phone|email', section_name):
        return "Contact"
    else:
        return "Other"

def is_job_title(text: str) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    Check if text is a job title
    
    Args:
        text: Text to check
        
    Returns:
        Tuple of (is_job_title, extracted_info)
    """
    # Common patterns for job titles
    job_title_patterns = [
        r'(?i)(^|\n)([A-Z][A-Za-z\s]+)\s+[-–|]\s+([A-Za-z0-9\s,]+)',  # Job Title - Company
        r'(?i)(^|\n)([A-Za-z\s]+)\s+at\s+([A-Za-z0-9\s,]+)',         # Job Title at Company
        r'(?i)(^|\n)([A-Z][A-Za-z\s]+),\s+([A-Za-z0-9\s,]+)',        # Job Title, Company
        r'(?i)(^|\n)([A-Za-z\s]+)\s+\(([A-Za-z0-9\s,]+)\)',          # Job Title (Company)
    ]
    
    for pattern in job_title_patterns:
        match = re.search(pattern, text.strip())
        if match:
            # Different patterns have different group structures
            if len(match.groups()) >= 3:
                return True, {
                    "title": match.groups()[1].strip(),
                    "company": match.groups()[2].strip()
                }
    
    return False, None

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data as JSON file
    
    Args:
        data: Data to save
        filepath: Path to save to
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {str(e)}")
        raise

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to load from
        
    Returns:
        Loaded data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON: {str(e)}")
        raise

def extract_contact_info(text: str) -> Dict[str, str]:
    """
    Extract contact information from text
    
    Args:
        text: Text to extract from
        
    Returns:
        Dictionary of contact information
    """
    contact_info = {}
    
    # Email
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    if email_match:
        contact_info['email'] = email_match.group(0)
    
    # Phone
    phone_match = re.search(r'(?:\+\d{1,3}\s?)?(?:\(\d{3}\)\s?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4}', text)
    if phone_match:
        contact_info['phone'] = phone_match.group(0)
    
    # LinkedIn
    linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text)
    if linkedin_match:
        contact_info['linkedin'] = linkedin_match.group(0)
    
    # Website
    website_match = re.search(r'https?://(?:www\.)?[\w.-]+\.[a-zA-Z]{2,}(?:/[\w./-]*)?', text)
    if website_match and 'linkedin.com' not in website_match.group(0):
        contact_info['website'] = website_match.group(0)
    
    # Location (simple heuristic)
    location_patterns = [
        r'(?:^|\n)([A-Za-z\s]+,\s*[A-Z]{2}(?:,\s*[A-Za-z\s]+)?)',  # City, ST format
        r'(?:^|\n)([A-Za-z\s]+,\s*[A-Za-z\s]+)'                    # City, Country format
    ]
    
    for pattern in location_patterns:
        location_match = re.search(pattern, text)
        if location_match:
            contact_info['location'] = location_match.group(1).strip()
            break
    
    return contact_info

# Testing utilities
if __name__ == "__main__":
    # Test clean_text
    test_text = "This is a test\nwith   multiple spaces\nand line breaks."
    print(f"Cleaned text: {clean_text(test_text)}")
    
    # Test detect_section_title
    test_titles = ["WORK EXPERIENCE", "Education", "Technical Skills:", "Random Text"]
    for title in test_titles:
        print(f"'{title}' is a section title: {detect_section_title(title)}")
    
    # Test extract_bullet_points
    bullet_text = "• First point\n• Second point\n- Third point"
    print(f"Extracted bullet points: {extract_bullet_points(bullet_text)}")
    
    # Test is_job_title
    test_job_titles = [
        "Software Engineer - Google",
        "Data Scientist at Microsoft",
        "Project Manager, ABC Corporation",
        "Regular text"
    ]
    
    for title in test_job_titles:
        is_job, info = is_job_title(title)
        if is_job:
            print(f"'{title}' is a job title: Title={info['title']}, Company={info['company']}")
        else:
            print(f"'{title}' is not a job title")
