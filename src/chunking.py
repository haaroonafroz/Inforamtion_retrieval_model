import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import os
import json
import logging
import torch
from PIL import Image
import cv2
import pdfplumber
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast
)

from src.config import load_config, get_section_name

# Configure logging
logger = logging.getLogger(__name__)

class ResumeChunker:
    """
    A chunker specifically designed for resumes/CVs that uses a hybrid approach:
    - Identifies main sections based on headings
    - Creates hierarchical chunks with parent-child relationships
    - Preserves structure of the document
    - Uses layout information for improved structure detection
    """
    
    # Common patterns for job titles/positions (sub-sections)
    JOB_TITLE_PATTERNS = [
        r"(?i)(^|\n)([A-Z][A-Za-z\s]+)\s+[-–|]\s+([A-Za-z0-9\s,]+)",  # Job Title - Company
        r"(?i)(^|\n)([A-Za-z\s]+)\s+at\s+([A-Za-z0-9\s,]+)",         # Job Title at Company
        r"(?i)(^|\n)([A-Z][A-Za-z\s]+),\s+([A-Za-z0-9\s,]+)",        # Job Title, Company
    ]
    
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base", use_layout_model: bool = True):
        """
        Initialize the resume chunker with layout-aware capabilities
        
        Args:
            model_name: The LayoutLMv3 model to use
            use_layout_model: Whether to use the LayoutLM model for improved layout detection
        """
        # Get config (used for section mappings)
        self.config = load_config()
        
        # Get section patterns from config and compile for efficiency
        self.section_patterns = []
        for section, patterns in self.config.get("section_mappings", {}).items():
            for pattern in patterns:
                if pattern.startswith("^") or pattern.startswith("(?i)^"):
                    # Already a proper regex pattern
                    self.section_patterns.append(re.compile(pattern))
                else:
                    # Make it a standalone word pattern
                    self.section_patterns.append(re.compile(f"(?i)^{pattern}$"))
        
        # Compile job title patterns for efficiency
        self.job_title_patterns = [re.compile(pattern) for pattern in self.JOB_TITLE_PATTERNS]
        
        # Add more specific job title patterns for common formats
        self.job_title_patterns.extend([
            # Job Title pattern with date range
            re.compile(r"(?i)(^|\n)([A-Za-z\s&]+)[\s\n]+([A-Za-z0-9\s,.&]+)[\s\n]+([\d]{1,2}/[\d]{4}|[\d]{4}|[A-Za-z]+\s*[\d]{4})\s*[-–]\s*([\d]{1,2}/[\d]{4}|[\d]{4}|[A-Za-z]+\s*[\d]{4}|[Pp]resent)"),
            # More specific pattern for "Job Title: Company"
            re.compile(r"(?i)(^|\n)([A-Za-z\s&]+):\s*([A-Za-z0-9\s,.&]+)"),
            # Pattern for titles like "Intern: Machine Learning Engineer & Data Analyst"
            re.compile(r"(?i)(^|\n)([A-Za-z\s&]+):\s*([A-Za-z\s&]+)")
        ])
        
        # Initialize layout-aware components if enabled
        self.use_layout_model = use_layout_model
        
        if self.use_layout_model:
            logger.info(f"Initializing layout-aware resume chunker with model: {model_name}")
            
            try:
                self.model_name = model_name
                self.processor = LayoutLMv3Processor.from_pretrained(model_name)
                self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_name)
                self.feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(model_name)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {self.device}")
                logger.info("LayoutLMv3 model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LayoutLMv3 model: {str(e)}")
                self.use_layout_model = False
                logger.warning("Falling back to standard resume chunking without layout model")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def extract_text_with_layout(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with layout information including:
        - Font size
        - Bold status
        - Position on page
        - Block structure
        """
        doc = fitz.open(pdf_path)
        blocks = []
        
        for page_num, page in enumerate(doc):
            # Get blocks which preserve more layout information
            page_dict = page.get_text("dict")
            for block in page_dict["blocks"]:
                if "lines" not in block:
                    continue
                    
                block_text = ""
                max_font_size = 0
                is_bold = False
                
                # Process each line in the block
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                        # Track maximum font size and bold status
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
                        if "bold" in span["font"].lower():
                            is_bold = True
                
                # Add block with metadata
                if block_text.strip():
                    blocks.append({
                        "text": block_text.strip(),
                        "font_size": max_font_size,
                        "is_bold": is_bold,
                        "page": page_num + 1,
                        "bbox": block["bbox"],  # x0, y0, x1, y1 coordinates
                    })
        
        return blocks
    
    def is_section_heading(self, text_line: str, font_size: Optional[float] = None, is_bold: bool = False) -> bool:
        """
        Check if a text line is a main section heading.
        
        Criteria:
        1. Matches one of the section patterns
        2. Optional: Has larger font or is bold (if layout info available)
        """
        # Check regex patterns first
        for pattern in self.section_patterns:
            if pattern.search(text_line.strip()):
                return True
        
        # If layout info available, use additional heuristics
        if font_size is not None and is_bold and len(text_line.strip()) < 30:
            # Short, bold, large font text is likely a heading
            return True
            
        return False
    
    def is_job_title(self, text_line: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        """Check if a text line is a job title/sub-heading and extract components."""
        # Try to match common job title patterns
        for pattern in self.job_title_patterns:
            match = pattern.search(text_line.strip())
            if match:
                # Different patterns have different group structures
                groups = match.groups()
                
                # Pattern with date range (groups contain date info)
                if len(groups) >= 5:
                    return True, {
                        "title": groups[1].strip(),
                        "company": groups[2].strip(),
                        "start_date": groups[3].strip(),
                        "end_date": groups[4].strip()
                    }
                # Standard pattern with title and company
                elif len(groups) >= 3:
                    return True, {
                        "title": groups[1].strip(),
                        "company": groups[2].strip()
                    }
                
        # Special case to improve "title at company" pattern with fixes for the "** at **" issue
        at_match = re.search(r'([^:]+)\s+at\s+([^:]+)', text_line.strip())
        if at_match and len(at_match.groups()) >= 2:
            title = at_match.group(1).strip()
            company = at_match.group(2).strip()
            
            # Only match if both title and company are meaningful (not empty, not just "at")
            if title and company and title != "at" and company != "at":
                return True, {
                    "title": title,
                    "company": company
                }
                
        # Special case for lines with dates that might be job entries
        date_match = re.search(r'(.+?)(\d{1,2}/\d{4}|\d{4})\s*[-–]\s*(\d{1,2}/\d{4}|\d{4}|[Pp]resent)', text_line.strip())
        if date_match:
            title_company = date_match.group(1).strip()
            # Try to split title and company if possible
            if ':' in title_company:
                parts = title_company.split(':', 1)
                return True, {
                    "title": parts[0].strip(),
                    "company": parts[1].strip() if len(parts) > 1 else "",
                    "start_date": date_match.group(2).strip(),
                    "end_date": date_match.group(3).strip()
                }
            else:
                # If we can't split, use the whole thing as title
                return True, {
                    "title": title_company,
                    "company": "",
                    "start_date": date_match.group(2).strip(),
                    "end_date": date_match.group(3).strip()
                }
        
        return False, None
    
    def extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text content."""
        # Split on common bullet point markers
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
                # Make sure there are spaces between words in bullet points
                bullet_points.extend([self._ensure_word_spacing(match) for match in matches])
                # Remove matched content for next pattern
                remaining_text = re.sub(pattern, '', remaining_text)
        
        # If no bullet points found but there are line breaks, treat lines as points
        if not bullet_points and '\n' in text:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # Filter out very short lines (likely not content)
            bullet_points = [self._ensure_word_spacing(line) for line in lines if len(line) > 10]
        
        return bullet_points
    
    def _ensure_word_spacing(self, text: str) -> str:
        """
        Ensure proper spacing between words by adding spaces where needed.
        Fixes the issue of words running together in extracted text.
        """
        # Look for camelCase or PascalCase patterns and add spaces
        # This handles cases like "AutomatedthereportingsystembyidentifyingdatapatternsanddiscrepanciesusingPythonscriptsandSQL"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Look for lowercase followed by uppercase letter and add space
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix spaces around punctuation if needed
        text = re.sub(r'([.:,;])([a-zA-Z])', r'\1 \2', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def chunk_resume(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk the resume text using a hybrid approach:
        - Identify main sections
        - Create hierarchical chunks with parent-child relationships
        """
        # Split text into lines for analysis
        lines = text.split('\n')
        
        chunks = []
        current_section = None
        current_subsection = None
        buffer = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Check if this is a main section heading
            if self.is_section_heading(line):
                # If we have content in buffer, save it to the previous chunk
                if buffer and current_section:
                    # Add content to previous section or subsection
                    if current_subsection:
                        current_subsection["content"] = '\n'.join(buffer)
                    else:
                        current_section["content"] = '\n'.join(buffer)
                    buffer = []
                
                # Create a new section
                current_section = {
                    "type": "section",
                    "heading": line,
                    "content": "",
                    "subsections": [],
                    "metadata": {
                        "line_number": i,
                        "normalized_heading": self.normalize_section_heading(line)
                    }
                }
                current_subsection = None
                chunks.append(current_section)
                
            # Check if this is a job title/subsection
            elif current_section:
                is_job, job_info = self.is_job_title(line)
                if is_job and job_info:
                    # If we have content in buffer, save it
                    if buffer:
                        if current_subsection:
                            current_subsection["content"] = '\n'.join(buffer)
                        else:
                            current_section["content"] = '\n'.join(buffer)
                        buffer = []
                    
                    # Create a new subsection
                    current_subsection = {
                        "type": "subsection",
                        "heading": line,
                        "title": job_info.get("title", ""),
                        "company": job_info.get("company", ""),
                        "content": "",
                        "metadata": {
                            "line_number": i
                        }
                    }
                    current_section["subsections"].append(current_subsection)
                else:
                    # Regular content line, add to buffer
                    buffer.append(line)
        
        # Don't forget to add the last buffer content
        if buffer:
            if current_subsection:
                current_subsection["content"] = '\n'.join(buffer)
            elif current_section:
                current_section["content"] = '\n'.join(buffer)
        
        return chunks
    
    def chunk_resume_with_layout(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk the resume using layout information
        
        Args:
            blocks: List of text blocks with layout metadata
        """
        chunks = []
        current_section = None
        current_subsection = None
        buffer = []
        
        # First pass: find average font size to identify headings
        font_sizes = [block["font_size"] for block in blocks if block["font_size"] > 0]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        heading_threshold = avg_font_size * 1.2  # 20% larger than average
        
        for i, block in enumerate(blocks):
            text = block["text"].strip()
            if not text:  # Skip empty blocks
                continue
            
            # Check if this is a section heading by layout and content
            is_heading = (
                block["font_size"] >= heading_threshold or 
                block["is_bold"] or 
                self.is_section_heading(text)
            )
            
            if is_heading:
                # Process previous buffer if exists
                if buffer and current_section:
                    content = '\n'.join([b["text"] for b in buffer])
                    
                    if current_subsection:
                        current_subsection["content"] = content
                        current_subsection["bullet_points"] = self.extract_bullet_points(content)
                    else:
                        current_section["content"] = content
                        current_section["bullet_points"] = self.extract_bullet_points(content)
                    
                    buffer = []
                
                # Create new section
                current_section = {
                    "type": "section",
                    "heading": text,
                    "content": "",
                    "bullet_points": [],
                    "subsections": [],
                    "metadata": {
                        "page": block["page"],
                        "bbox": block["bbox"],
                        "font_size": block["font_size"],
                        "is_bold": block["is_bold"],
                        "normalized_heading": self.normalize_section_heading(text)
                    }
                }
                current_subsection = None
                chunks.append(current_section)
                
            # Check if this is a subsection/job title
            elif current_section:
                is_job, job_info = self.is_job_title(text)
                
                if is_job and job_info:
                    # Process previous buffer
                    if buffer:
                        content = '\n'.join([b["text"] for b in buffer])
                        
                        if current_subsection:
                            current_subsection["content"] = content
                            current_subsection["bullet_points"] = self.extract_bullet_points(content)
                        else:
                            current_section["content"] = content
                            current_section["bullet_points"] = self.extract_bullet_points(content)
                        
                        buffer = []
                    
                    # Create new subsection
                    current_subsection = {
                        "type": "subsection",
                        "heading": text,
                        "title": job_info.get("title", ""),
                        "company": job_info.get("company", ""),
                        "content": "",
                        "bullet_points": [],
                        "metadata": {
                            "page": block["page"],
                            "bbox": block["bbox"],
                            "font_size": block["font_size"],
                            "is_bold": block["is_bold"]
                        }
                    }
                    current_section["subsections"].append(current_subsection)
                else:
                    # Regular content, add to buffer
                    buffer.append(block)
        
        # Process the last buffer
        if buffer:
            content = '\n'.join([b["text"] for b in buffer])
            
            if current_subsection:
                current_subsection["content"] = content
                current_subsection["bullet_points"] = self.extract_bullet_points(content)
            elif current_section:
                current_section["content"] = content
                current_section["bullet_points"] = self.extract_bullet_points(content)
        
        return chunks
    
    def normalize_section_heading(self, heading: str) -> str:
        """
        Normalize section heading to standard name using config mappings
        
        Args:
            heading: Section heading text
            
        Returns:
            Normalized section name
        """
        # Use the function from config.py
        return get_section_name(heading)
    
    def flatten_chunks_for_embedding(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert hierarchical chunks to flat chunks for embedding and retrieval.
        Each chunk gets metadata about its hierarchy.
        """
        flat_chunks = []
        
        for section in chunks:
            # Convert bbox tuple to string if it exists
            bbox = section.get('metadata', {}).get('bbox')
            bbox_str = str(bbox) if bbox is not None else None
            
            # Add the main section as a chunk
            section_chunk = {
                "text": f"{section['heading']}\n{section['content']}",
                "metadata": {
                    "type": "section",
                    "heading": section['heading'],
                    "normalized_heading": section.get('metadata', {}).get('normalized_heading', 'Other'),
                    "page": section.get('metadata', {}).get('page'),
                    "bbox_str": bbox_str  # Store as string instead of tuple
                }
            }
            flat_chunks.append(section_chunk)
            
            # Add each subsection as a separate chunk
            for subsection in section.get("subsections", []):
                # Convert bbox tuple to string if it exists
                subsection_bbox = subsection.get('metadata', {}).get('bbox')
                subsection_bbox_str = str(subsection_bbox) if subsection_bbox is not None else None
                
                subsection_chunk = {
                    "text": f"{subsection['heading']}\n{subsection['content']}",
                    "metadata": {
                        "type": "subsection",
                        "parent_section": section['heading'],
                        "parent_normalized": section.get('metadata', {}).get('normalized_heading', 'Other'),
                        "heading": subsection['heading'],
                        "title": subsection.get('title', ''),
                        "company": subsection.get('company', ''),
                        "page": subsection.get('metadata', {}).get('page'),
                        "bbox_str": subsection_bbox_str  # Store as string instead of tuple
                    }
                }
                flat_chunks.append(subsection_chunk)
        
        return flat_chunks
    
    def convert_to_output_format(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert chunks to the desired output JSON format per the requirements
        
        Example:
        {
          "Experience": {
            "Software Engineer - Google": [
              "Developed scalable backend services in Python",
              "Improved API performance by 30% using Redis caching"
            ]
          },
          "Education": { ... }
        }
        """
        output = {}
        
        for section in chunks:
            normalized_heading = section.get('metadata', {}).get('normalized_heading') or self.normalize_section_heading(section['heading'])
            
            # Initialize the section if not exists
            if normalized_heading not in output:
                output[normalized_heading] = {}
            
            # Handle subsections
            if section.get("subsections"):
                for subsection in section["subsections"]:
                    # Create key from title and company
                    title = subsection.get("title", "")
                    company = subsection.get("company", "")
                    key = f"{title} - {company}" if title and company else subsection["heading"]
                    
                    # Get bullet points or create from content
                    if "bullet_points" in subsection and subsection["bullet_points"]:
                        bullet_points = subsection["bullet_points"]
                    else:
                        bullet_points = self.extract_bullet_points(subsection["content"])
                        
                    # Use content lines if no bullet points found
                    if not bullet_points and subsection["content"]:
                        bullet_points = [line.strip() for line in subsection["content"].split('\n') if line.strip()]
                    
                    # Add to output
                    output[normalized_heading][key] = bullet_points
            else:
                # Handle sections without subsections (like Skills)
                if normalized_heading in ["Skills", "Languages", "Interests"]:
                    # For these sections, extract as a flat list
                    if "bullet_points" in section and section["bullet_points"]:
                        output[normalized_heading] = section["bullet_points"]
                    else:
                        bullet_points = self.extract_bullet_points(section["content"])
                        if bullet_points:
                            output[normalized_heading] = bullet_points
                        else:
                            # Split by commas if no bullet points found
                            skills = [s.strip() for s in re.split(r',|\n', section["content"]) if s.strip()]
                            output[normalized_heading] = skills
                else:
                    # For other sections without subsections, use the content as is
                    key = section["heading"]
                    if "bullet_points" in section and section["bullet_points"]:
                        output[normalized_heading][key] = section["bullet_points"]
                    else:
                        bullet_points = self.extract_bullet_points(section["content"])
                        if bullet_points:
                            output[normalized_heading][key] = bullet_points
                        else:
                            # Use content lines as bullet points
                            lines = [line.strip() for line in section["content"].split('\n') if line.strip()]
                            output[normalized_heading][key] = lines
        
        return output
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF resume/CV and return structured chunked data
        Uses layout-aware processing if enabled, falls back to standard processing otherwise
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        if self.use_layout_model:
            # Layout-aware processing using pdfplumber and layout information
            try:
                # 1. Extract text elements with layout information
                elements = self.extract_text_info_from_pdf(pdf_path)
                
                # 2. Group elements into lines
                lines = self.group_elements_into_lines(elements)
                
                # 3. Classify line roles (heading, subheading, content)
                classified_lines = self.classify_line_roles(lines)
                
                # 4. Build hierarchical structure
                structured_data = self.build_hierarchical_structure(classified_lines)
                
                # Create flattened chunks suitable for embedding
                flat_chunks = self.create_flat_chunks_from_structured_data(structured_data, pdf_path)
                
                # Convert to hierarchical chunks format
                hierarchical_chunks = self.create_hierarchical_chunks_from_structured_data(structured_data)
                
                return {
                    "hierarchical_chunks": hierarchical_chunks,
                    "flat_chunks": flat_chunks,
                    "output_json": structured_data
                }
                
            except Exception as e:
                logger.error(f"Layout-aware processing failed: {str(e)}. Falling back to standard processing.")
                # Fall back to standard processing below
        
        # Standard resume chunker processing using PyMuPDF
        # Extract text with layout information
        blocks = self.extract_text_with_layout(pdf_path)
        
        # Create hierarchical chunks
        hierarchical_chunks = self.chunk_resume_with_layout(blocks)
        
        # Create flattened chunks suitable for embedding
        flat_chunks = self.flatten_chunks_for_embedding(hierarchical_chunks)
        
        # Convert to output format
        output_json = self.convert_to_output_format(hierarchical_chunks)
        
        return {
            "hierarchical_chunks": hierarchical_chunks,
            "flat_chunks": flat_chunks,
            "output_json": output_json
        }
    
    def create_flat_chunks_from_structured_data(self, structured_data: Dict[str, Any], pdf_path: str) -> List[Dict[str, Any]]:
        """
        Create flat chunks from structured data for vector store
        
        Args:
            structured_data: Structured data from layout-aware processing
            pdf_path: Path to the PDF file
            
        Returns:
            List of flat chunks
        """
        flat_chunks = []
        
        # Process experience sections
        if "experience" in structured_data:
            for idx, job in enumerate(structured_data["experience"]):
                # Handle description properly
                if isinstance(job.get("description", ""), str):
                    job_text = f"{job.get('title', '')} at {job.get('company', '')}\n{job.get('description', '')}"
                else:
                    description_str = str(job.get("description", ""))
                    job_text = f"{job.get('title', '')} at {job.get('company', '')}\n{description_str}"
                
                flat_chunks.append({
                    "text": job_text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "type": "subsection",
                        "normalized_heading": "Experience",
                        "index": idx
                    }
                })
        
        # Process education sections
        if "education" in structured_data:
            for idx, edu in enumerate(structured_data["education"]):
                # Handle description properly
                if isinstance(edu.get("description", ""), str):
                    edu_text = f"{edu.get('degree', '')} - {edu.get('institution', '')}\n{edu.get('description', '')}"
                else:
                    description_str = str(edu.get("description", ""))
                    edu_text = f"{edu.get('degree', '')} - {edu.get('institution', '')}\n{description_str}"
                
                flat_chunks.append({
                    "text": edu_text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "type": "subsection",
                        "normalized_heading": "Education",
                        "index": idx
                    }
                })
        
        # Process skills
        if "skills" in structured_data:
            skills_text = ", ".join(str(skill) for skill in structured_data["skills"])
            flat_chunks.append({
                "text": skills_text,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "type": "section",
                    "normalized_heading": "Skills",
                    "index": 0
                }
            })
        
        # Process projects
        if "projects" in structured_data:
            for idx, proj in enumerate(structured_data["projects"]):
                # Make sure we're handling the description properly regardless of type
                if isinstance(proj.get("description", []), list):
                    # If it's a list, convert any non-string items to strings
                    description_items = [str(item) for item in proj.get("description", [])]
                    proj_text = f"{proj.get('title', '')}\n{' '.join(description_items)}"
                else:
                    # If it's a string or other type, convert it to string
                    proj_text = f"{proj.get('title', '')}\n{str(proj.get('description', ''))}"
                
                flat_chunks.append({
                    "text": proj_text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "type": "subsection",
                        "normalized_heading": "Projects",
                        "index": idx
                    }
                })
        
        # Process summary
        if "summary" in structured_data and structured_data["summary"]:
            # Make sure summary is a string
            if not isinstance(structured_data["summary"], str):
                summary_text = str(structured_data["summary"])
            else:
                summary_text = structured_data["summary"]
            
            flat_chunks.append({
                "text": summary_text,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "type": "section",
                    "normalized_heading": "Summary",
                    "index": 0
                }
            })
            
        return flat_chunks
    
    def create_hierarchical_chunks_from_structured_data(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks from structured data for the output format
        
        Args:
            structured_data: Structured data from layout-aware processing
            
        Returns:
            List of hierarchical chunks
        """
        hierarchical_chunks = []
        
        # Create the Experience section
        if "experience" in structured_data and structured_data["experience"]:
            experience_section = {
                "type": "section",
                "heading": "Experience",
                "content": "",
                "subsections": [],
                "metadata": {
                    "normalized_heading": "Experience"
                }
            }
            
            for job in structured_data["experience"]:
                subsection = {
                    "type": "subsection",
                    "heading": f"{job.get('title', '')} at {job.get('company', '')}",
                    "content": job.get("description", ""),
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "metadata": {
                        "normalized_heading": "Experience"
                    }
                }
                experience_section["subsections"].append(subsection)
            
            hierarchical_chunks.append(experience_section)
        
        # Create the Education section
        if "education" in structured_data and structured_data["education"]:
            education_section = {
                "type": "section",
                "heading": "Education",
                "content": "",
                "subsections": [],
                "metadata": {
                    "normalized_heading": "Education"
                }
            }
            
            for edu in structured_data["education"]:
                subsection = {
                    "type": "subsection",
                    "heading": f"{edu.get('degree', '')} - {edu.get('institution', '')}",
                    "content": edu.get("description", ""),
                    "metadata": {
                        "normalized_heading": "Education"
                    }
                }
                education_section["subsections"].append(subsection)
            
            hierarchical_chunks.append(education_section)
        
        # Create the Skills section
        if "skills" in structured_data and structured_data["skills"]:
            skills_section = {
                "type": "section",
                "heading": "Skills",
                "content": ", ".join(structured_data["skills"]),
                "subsections": [],
                "bullet_points": structured_data["skills"],
                "metadata": {
                    "normalized_heading": "Skills"
                }
            }
            hierarchical_chunks.append(skills_section)
        
        # Create the Projects section
        if "projects" in structured_data and structured_data["projects"]:
            projects_section = {
                "type": "section",
                "heading": "Projects",
                "content": "",
                "subsections": [],
                "metadata": {
                    "normalized_heading": "Projects"
                }
            }
            
            for proj in structured_data["projects"]:
                content = ""
                if isinstance(proj.get("description", []), list):
                    content = "\n".join(proj["description"])
                else:
                    content = str(proj.get("description", ""))
                    
                subsection = {
                    "type": "subsection",
                    "heading": proj.get("title", ""),
                    "content": content,
                    "metadata": {
                        "normalized_heading": "Projects"
                    }
                }
                projects_section["subsections"].append(subsection)
            
            hierarchical_chunks.append(projects_section)
        
        # Create the Summary section
        if "summary" in structured_data and structured_data["summary"]:
            summary_section = {
                "type": "section",
                "heading": "Summary",
                "content": structured_data["summary"],
                "subsections": [],
                "metadata": {
                    "normalized_heading": "Summary"
                }
            }
            hierarchical_chunks.append(summary_section)
        
        return hierarchical_chunks
    
    def extract_text_info_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text with layout information from PDF using pdfplumber
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text elements with layout information
        """
        logger.info(f"Extracting text info from PDF: {pdf_path}")
        elements = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                # Extract words with layout information
                words = page.extract_words(
                    x_tolerance=3, 
                    y_tolerance=3, 
                    keep_blank_chars=False
                )
                
                # Process each word
                for word in words:
                    elements.append({
                        "text": word["text"],
                        "x0": word["x0"],
                        "x1": word["x1"],
                        "top": word["top"],
                        "bottom": word["bottom"],
                        "width": word["x1"] - word["x0"],
                        "height": word["bottom"] - word["top"],
                        "page": page_number
                    })
                
                # Get character-level details for font information
                chars = page.chars
                
                # Group characters by word position to get font size and style
                for word in elements:
                    if word["page"] == page_number:
                        # Find characters within the word's bounding box
                        word_chars = [c for c in chars if 
                                     c["x0"] >= word["x0"] - 2 and
                                     c["x1"] <= word["x1"] + 2 and
                                     c["top"] >= word["top"] - 2 and
                                     c["bottom"] <= word["bottom"] + 2]
                        
                        if word_chars:
                            # Calculate average font size
                            font_sizes = [c.get("size", 0) for c in word_chars]
                            word["font_size"] = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                            
                            # Check if any character is bold
                            word["is_bold"] = any("bold" in c.get("fontname", "").lower() for c in word_chars)
                        else:
                            word["font_size"] = 0
                            word["is_bold"] = False
        
        return elements
    
    def group_elements_into_lines(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group elements (words) into lines based on vertical position
        
        Args:
            elements: List of elements with layout information
            
        Returns:
            List of lines with grouped words
        """
        # Sort elements by page and vertical position
        elements.sort(key=lambda x: (x["page"], x["top"]))
        
        lines = []
        current_line = []
        current_page = None
        current_top = None
        
        for element in elements:
            if current_page is None:
                # First element
                current_page = element["page"]
                current_top = element["top"]
                current_line.append(element)
            elif (element["page"] != current_page or 
                  abs(element["top"] - current_top) > max(element["height"] * 0.5, 5)):
                # New page or new line (vertical gap is significant)
                if current_line:
                    # Sort words in line by horizontal position
                    current_line.sort(key=lambda x: x["x0"])
                    
                    # Create line object
                    line_text = " ".join([e["text"] for e in current_line])
                    
                    # Use highest font size and bold status in the line
                    max_font_size = max([e.get("font_size", 0) for e in current_line])
                    is_bold = any([e.get("is_bold", False) for e in current_line])
                    
                    # Get bounding box of entire line
                    x0 = min([e["x0"] for e in current_line])
                    x1 = max([e["x1"] for e in current_line])
                    top = min([e["top"] for e in current_line])
                    bottom = max([e["bottom"] for e in current_line])
                    
                    # Check if this line might be a date range
                    contains_date = bool(re.search(r'(\d{1,2}/\d{4}|\d{4})\s*[-–]\s*(\d{1,2}/\d{4}|\d{4}|[Pp]resent)', line_text))
                    
                    # Check if this might be a job title or company name
                    # Look for indicators like "Engineer", "Developer", "Analyst", etc.
                    job_title_words = ["engineer", "developer", "analyst", "manager", "director", 
                                      "coordinator", "specialist", "consultant", "intern", "trainee"]
                    might_be_job_title = any(word.lower() in line_text.lower() for word in job_title_words)
                    
                    lines.append({
                        "text": line_text,
                        "page": current_page,
                        "font_size": max_font_size,
                        "is_bold": is_bold,
                        "x0": x0,
                        "x1": x1,
                        "top": top,
                        "bottom": bottom,
                        "words": current_line,
                        "contains_date": contains_date,
                        "might_be_job_title": might_be_job_title
                    })
                
                # Start new line
                current_line = [element]
                current_page = element["page"]
                current_top = element["top"]
            else:
                # Same line, add element
                current_line.append(element)
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda x: x["x0"])
            line_text = " ".join([e["text"] for e in current_line])
            max_font_size = max([e.get("font_size", 0) for e in current_line])
            is_bold = any([e.get("is_bold", False) for e in current_line])
            x0 = min([e["x0"] for e in current_line])
            x1 = max([e["x1"] for e in current_line])
            top = min([e["top"] for e in current_line])
            bottom = max([e["bottom"] for e in current_line])
            
            # Check if this line might be a date range
            contains_date = bool(re.search(r'(\d{1,2}/\d{4}|\d{4})\s*[-–]\s*(\d{1,2}/\d{4}|\d{4}|[Pp]resent)', line_text))
            
            # Check if this might be a job title or company name
            job_title_words = ["engineer", "developer", "analyst", "manager", "director", 
                              "coordinator", "specialist", "consultant", "intern", "trainee"]
            might_be_job_title = any(word.lower() in line_text.lower() for word in job_title_words)
            
            lines.append({
                "text": line_text,
                "page": current_page,
                "font_size": max_font_size,
                "is_bold": is_bold,
                "x0": x0,
                "x1": x1,
                "top": top,
                "bottom": bottom,
                "words": current_line,
                "contains_date": contains_date,
                "might_be_job_title": might_be_job_title
            })
        
        return lines
        
    def classify_line_roles(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify each line as a heading, subheading, or content
        based on layout features and content
        
        Args:
            lines: List of text lines with layout information
            
        Returns:
            Lines with role classification added
        """
        if not lines:
            return []
        
        # Calculate statistics for font sizes to identify headings and subheadings
        font_sizes = [line["font_size"] for line in lines if line["font_size"] > 0]
        if not font_sizes:
            return lines
        
        avg_font_size = sum(font_sizes) / len(font_sizes)
        
        # Set thresholds for headings and subheadings
        heading_threshold = avg_font_size * 1.3  # 30% larger than average
        subheading_threshold = avg_font_size * 1.1  # 10% larger than average
        
        # Previous line role for context
        prev_role = None
        
        for i, line in enumerate(lines):
            # Initial role based on font size
            if line["font_size"] >= heading_threshold or (line["is_bold"] and line["font_size"] > avg_font_size):
                line["role"] = "heading"
            elif line["font_size"] >= subheading_threshold or line["is_bold"]:
                line["role"] = "subheading"
            else:
                line["role"] = "content"
            
            # Check if text matches section heading patterns
            for pattern in self.section_patterns:
                if pattern.search(line["text"].strip()):
                    line["role"] = "heading"
                    break
            
            # Check if text matches job title patterns
            is_job, _ = self.is_job_title(line["text"])
            if is_job:
                line["role"] = "subheading"
            
            # Use the job title heuristics from group_elements_into_lines
            if line.get("might_be_job_title", False) and line.get("contains_date", False):
                line["role"] = "subheading"
            
            # Use context: first line is likely a heading (name)
            if i == 0 and line["role"] != "heading":
                line["role"] = "heading"
                
            # Use context: bullet points are content
            if line["text"].strip().startswith(("•", "-", "✓", "✔", "■", "○", "▪", "*")):
                line["role"] = "content"
                
            # Use context: date ranges in subheadings (check for year patterns)
            if re.search(r'\b(19|20)\d{2}\b.*\b(19|20)\d{2}|present\b', line["text"], re.IGNORECASE):
                if line["role"] != "heading":
                    line["role"] = "subheading"
            
            # Use context: short line after heading is likely a subheading
            if prev_role == "heading" and len(line["text"]) < 50 and line["role"] == "content":
                line["role"] = "subheading"
                
            prev_role = line["role"]
        
        return lines
    
    def is_contact_line(self, text: str) -> bool:
        """Check if a line contains contact information"""
        patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
            r'(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}',  # phone
            r'(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=|linkedin\.com/pub/)[A-Za-z0-9_-]+',  # linkedin
            r'(?:github\.com/)[A-Za-z0-9_-]+',  # github
        ]
        
        return any(re.search(pattern, text) for pattern in patterns)

    def build_hierarchical_structure(self, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a hierarchical structure from classified lines
        
        Args:
            lines: List of lines with role classification
            
        Returns:
            Hierarchical structure with sections, subsections, and content
        """
        structure = {}
        current_section = None
        current_subsection = None
        personal_info = {"name": "", "email": "", "phone": "", "title": "", "location": "", "socialLinks": {}}
        
        # Extract name from the first heading (usually the first line is the name)
        if lines and lines[0]["role"] == "heading":
            personal_info["name"] = lines[0]["text"].strip()
        
        # Process lines to build the hierarchy
        i = 0
        while i < len(lines):
            line = lines[i]
            text = line["text"].strip()
            if not text:
                i += 1
                continue
            
            # Handle headings (main sections)
            if line["role"] == "heading":
                # Special case for contact information
                if self.is_contact_line(text) and i < 5:  # Contact info usually at the top
                    # Extract email
                    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
                    if email_match:
                        personal_info["email"] = email_match.group(0)
                    
                    # Extract phone
                    phone_match = re.search(r'(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text)
                    if phone_match:
                        personal_info["phone"] = phone_match.group(0)
                    
                    # Extract LinkedIn
                    linkedin_match = re.search(r'(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=|linkedin\.com/pub/)[A-Za-z0-9_-]+', text)
                    if linkedin_match:
                        personal_info["socialLinks"]["linkedin"] = "https://www." + linkedin_match.group(0)
                    
                    # Extract GitHub
                    github_match = re.search(r'(?:github\.com/)[A-Za-z0-9_-]+', text)
                    if github_match:
                        personal_info["socialLinks"]["github"] = "https://www." + github_match.group(0)
                    
                    i += 1
                    continue
                
                # Normalize section name
                section_key = get_section_name(text)
                
                # Create section if it doesn't exist
                if section_key not in structure:
                    structure[section_key] = []
                
                current_section = section_key
                current_subsection = None
                i += 1
            
            # Handle job titles and subheadings
            elif line["role"] == "subheading" and current_section:
                # Create a new subsection object
                subsection = {
                    "text": text,
                    "content": []
                }
                
                # Try to extract job information for experience sections
                if current_section == "Experience":
                    # First, try to use our job title extractor
                    is_job, job_info = self.is_job_title(text)
                    if is_job and job_info:
                        if "title" in job_info:
                            subsection["title"] = job_info["title"]
                        if "company" in job_info:
                            subsection["company"] = job_info["company"]
                        if "start_date" in job_info:
                            subsection["startDate"] = job_info["start_date"]
                        if "end_date" in job_info:
                            subsection["endDate"] = job_info["end_date"]
                    else:
                        # Fallback approach: look for patterns in the text
                        # Check for "Job Title - Company" or "Job Title at Company" pattern
                        position_match = re.search(r'(.+?)(?:\s+-\s+|\s+at\s+)(.+)', text)
                        if position_match:
                            subsection["title"] = position_match.group(1).strip()
                            subsection["company"] = position_match.group(2).strip()
                        
                        # Check for "Job Title: Company" pattern
                        elif ":" in text:
                            parts = text.split(":", 1)
                            subsection["title"] = parts[0].strip()
                            if len(parts) > 1:
                                subsection["company"] = parts[1].strip()
                    
                    # Look for dates in this line or the next
                    date_line = text
                    if i + 1 < len(lines):
                        date_line += " " + lines[i + 1]["text"]
                    
                    # Extract date range if present and not already set
                    if not subsection.get("startDate") or not subsection.get("endDate"):
                        date_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s\w,.]+\d{4}|(?:\d{1,2}/\d{4}|\d{4}))\s*[-–]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s\w,.]+\d{4}|(?:\d{1,2}/\d{4}|\d{4})|[Pp]resent)', date_line)
                        if date_match:
                            subsection["startDate"] = date_match.group(1)
                            subsection["endDate"] = date_match.group(2)
                
                # Try to extract education information
                elif current_section == "Education":
                    # Check for degree and institution pattern
                    edu_match = re.search(r'(.+?)(?:\s+-\s+|\s+at\s+|\s*,\s*)(.+)', text)
                    if edu_match:
                        subsection["degree"] = edu_match.group(1).strip()
                        subsection["institution"] = edu_match.group(2).strip()
                    else:
                        subsection["degree"] = text
                    
                    # Look for dates in next line
                    if i + 1 < len(lines) and lines[i + 1]["role"] == "content":
                        date_match = re.search(r'((?:\d{1,2}/\d{4}|\d{4}))\s*[-–]\s*((?:\d{1,2}/\d{4}|\d{4})|[Pp]resent)', lines[i + 1]["text"])
                        if date_match:
                            subsection["startDate"] = date_match.group(1)
                            subsection["endDate"] = date_match.group(2)
                
                # Add subsection to current section
                structure[current_section].append(subsection)
                current_subsection = subsection
                i += 1
            
            # Handle content lines
            elif current_section:
                # If this is a skills section, handle differently
                if current_section == "Skills":
                    # Extract skills from bullet points or comma-separated list
                    skills = []
                    if "•" in text or "-" in text:
                        # Extract each bullet point as a skill
                        bullet_points = re.findall(r'[•\-]\s*([^•\-]+)', text)
                        skills.extend([self._ensure_word_spacing(bp.strip()) for bp in bullet_points if bp.strip()])
                    else:
                        # Split by commas
                        comma_skills = [self._ensure_word_spacing(s.strip()) for s in text.split(",") if s.strip()]
                        skills.extend(comma_skills)
                    
                    # Add skills to the list
                    if not structure["Skills"]:
                        structure["Skills"] = skills
                    else:
                        structure["Skills"].extend(skills)
                
                # For sections with subsections
                elif current_subsection:
                    # Add content to current subsection
                    # Make sure to preprocess text for better spacing
                    current_subsection["content"].append(self._ensure_word_spacing(text))
                
                # For sections without subsections (like summary)
                elif current_section:
                    # Ensure there's at least an empty list to append to
                    if not structure[current_section]:
                        structure[current_section] = []
                    
                    # If it's a list and the section itself is a subsection, add content
                    if isinstance(structure[current_section], list) and len(structure[current_section]) > 0:
                        if not structure[current_section][-1].get("content"):
                            structure[current_section][-1]["content"] = []
                        structure[current_section][-1]["content"].append(self._ensure_word_spacing(text))
                    # Otherwise just append directly to the section
                    else:
                        structure[current_section].append(self._ensure_word_spacing(text))
                
                i += 1
        
        # Post-process to match the required format
        result = {}
        
        # Process experience sections
        if "Experience" in structure:
            result["experience"] = []
            
            # Check each job entry for content before adding it
            for job in structure["Experience"]:
                # Skip entries that don't have both a title and description content
                has_title = job.get("title", "") != ""
                has_description = isinstance(job.get("content", []), list) and len(job.get("content", [])) > 0
                
                # Only include valid job entries with real content
                if has_title and has_description:
                    # Process each content line, ensuring bullet points are properly formatted
                    description_content = []
                    if isinstance(job.get("content", []), list):
                        for content_line in job.get("content", []):
                            # Check if it's a bullet point
                            if content_line.startswith(("•", "-", "✓", "✔", "■", "○", "▪", "*")):
                                # Extract the text after the bullet
                                bullet_text = re.sub(r'^[•\-✓✔■○▪\*]\s*', '', content_line)
                                description_content.append(bullet_text)
                            else:
                                description_content.append(content_line)
                    
                    # Only include entries with substantive content (not just empty strings)
                    if description_content and any(len(item.strip()) > 10 for item in description_content):
                        job_entry = {
                            "title": job.get("title", ""),
                            "company": job.get("company", ""),
                            "startDate": job.get("startDate", ""),
                            "endDate": job.get("endDate", ""),
                            "location": "",  # Could be extracted with more complex logic
                            "responsibilities": description_content  # Use only one field for content
                        }
                        result["experience"].append(job_entry)
        
        # Process education sections
        if "Education" in structure:
            result["education"] = []
            for edu in structure["Education"]:
                # Only include education entries with meaningful content
                if edu.get("degree", "") or edu.get("institution", ""):
                    edu_entry = {
                        "degree": edu.get("degree", ""),
                        "institution": edu.get("institution", ""),
                        "startDate": edu.get("startDate", ""),
                        "endDate": edu.get("endDate", ""),
                        "location": "",  # Could be extracted with more complex logic
                        "description": "\n".join(edu.get("content", [])) if isinstance(edu.get("content", []), list) else str(edu.get("content", ""))
                    }
                    result["education"].append(edu_entry)
        
        # Process skills section
        if "Skills" in structure:
            # Filter out any empty or very short skills (likely noise)
            result["skills"] = [str(skill) for skill in structure["Skills"] if len(str(skill).strip()) > 2]
        
        # Process projects section
        if "Projects" in structure:
            result["projects"] = []
            for proj in structure["Projects"]:
                # Only include projects with a title and some description
                has_title = len(proj.get("text", "").strip()) > 0
                has_content = isinstance(proj.get("content", []), list) and len(proj.get("content", [])) > 0
                
                if has_title and has_content:
                    # Process content into proper bullet points
                    description_content = []
                    if isinstance(proj.get("content", []), list):
                        for content_line in proj.get("content", []):
                            # Check if it's a bullet point
                            if content_line.startswith(("•", "-", "✓", "✔", "■", "○", "▪", "*")):
                                # Extract the text after the bullet
                                bullet_text = re.sub(r'^[•\-✓✔■○▪\*]\s*', '', content_line)
                                description_content.append(bullet_text)
                            else:
                                description_content.append(content_line)
                    
                    # Only include if there's meaningful description content
                    if description_content and any(len(item.strip()) > 10 for item in description_content):
                        proj_entry = {
                            "title": proj.get("text", ""),
                            "organization": "",
                            "startDate": proj.get("startDate", ""),
                            "endDate": proj.get("endDate", ""),
                            "link": "",  # Could extract from content with regex
                            "description": description_content
                        }
                        result["projects"].append(proj_entry)
        
        # Process certifications
        if "Certifications" in structure:
            result["certifications"] = []
            for cert in structure["Certifications"]:
                # Only include certifications with title
                if cert.get("text", "").strip():
                    cert_entry = {
                        "title": cert.get("text", ""),
                        "issuer": "",  # Could extract from content
                        "issueDate": "",
                        "expiryDate": "",
                        "link": "",
                        "description": "\n".join(cert.get("content", [])) if isinstance(cert.get("content", []), list) else str(cert.get("content", ""))
                    }
                    result["certifications"].append(cert_entry)
        
        # Add summary if present
        if "Summary" in structure:
            if isinstance(structure["Summary"], list):
                if all(isinstance(item, str) for item in structure["Summary"]):
                    result["summary"] = "\n".join(structure["Summary"])
                else:
                    # Try to extract text from more complex structure
                    summary_parts = []
                    for item in structure["Summary"]:
                        if isinstance(item, str):
                            summary_parts.append(item)
                        elif isinstance(item, dict) and "content" in item:
                            if isinstance(item["content"], list):
                                summary_parts.extend(item["content"])
                            else:
                                summary_parts.append(str(item["content"]))
                    result["summary"] = "\n".join(summary_parts)
            else:
                result["summary"] = str(structure["Summary"])
        else:
            result["summary"] = ""
        
        # Add personal info
        result["personalInfo"] = personal_info
        
        return result