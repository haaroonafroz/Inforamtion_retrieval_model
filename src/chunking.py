import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class ResumeChunker:
    """
    A chunker specifically designed for resumes/CVs that uses a hybrid approach:
    - Identifies main sections based on headings
    - Creates hierarchical chunks with parent-child relationships
    - Preserves structure of the document
    """
    
    # Common section headers found in resumes
    SECTION_PATTERNS = [
        r"(?i)^(professional\s+experience|experience|work\s+experience|employment(\s+history)?)$",
        r"(?i)^(education(\s+and\s+training)?)$",
        r"(?i)^(skills|technical\s+skills|core\s+competencies)$",
        r"(?i)^(projects|personal\s+projects)$",
        r"(?i)^(certifications|certificates)$",
        r"(?i)^(publications)$",
        r"(?i)^(languages)$",
        r"(?i)^(interests|hobbies)$",
        r"(?i)^(summary|profile|professional\s+summary|about\s+me)$",
        r"(?i)^(accomplishments|achievements)$",
        r"(?i)^(volunteer(\s+experience)?)$",
        r"(?i)^(references)$",
        r"(?i)^(contact(\s+information)?)$",
    ]
    
    # Common patterns for job titles/positions (sub-sections)
    JOB_TITLE_PATTERNS = [
        r"(?i)(^|\n)([A-Z][A-Za-z\s]+)\s+[-–|]\s+([A-Za-z0-9\s,]+)",  # Job Title - Company
        r"(?i)(^|\n)([A-Za-z\s]+)\s+at\s+([A-Za-z0-9\s,]+)",         # Job Title at Company
        r"(?i)(^|\n)([A-Z][A-Za-z\s]+),\s+([A-Za-z0-9\s,]+)",        # Job Title, Company
    ]
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.section_patterns = [re.compile(pattern) for pattern in self.SECTION_PATTERNS]
        self.job_title_patterns = [re.compile(pattern) for pattern in self.JOB_TITLE_PATTERNS]
    
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
        for pattern in self.job_title_patterns:
            match = pattern.search(text_line.strip())
            if match:
                # Different patterns have different group structures
                if len(match.groups()) >= 3:
                    return True, {
                        "title": match.groups()[1].strip(),
                        "company": match.groups()[2].strip()
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
                bullet_points.extend(matches)
                # Remove matched content for next pattern
                remaining_text = re.sub(pattern, '', remaining_text)
        
        # If no bullet points found but there are line breaks, treat lines as points
        if not bullet_points and '\n' in text:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # Filter out very short lines (likely not content)
            bullet_points = [line for line in lines if len(line) > 10]
        
        return bullet_points
    
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
        """Normalize section headings to standard categories."""
        heading_lower = heading.lower().strip()
        
        # Map to standard categories
        if re.search(r'experience|employment|work', heading_lower):
            return "Experience"
        elif re.search(r'education|academic|qualification', heading_lower):
            return "Education"
        elif re.search(r'skill|competenc|proficienc', heading_lower):
            return "Skills"
        elif re.search(r'project', heading_lower):
            return "Projects"
        elif re.search(r'certification|certificate', heading_lower):
            return "Certifications"
        elif re.search(r'publication', heading_lower):
            return "Publications"
        elif re.search(r'language', heading_lower):
            return "Languages"
        elif re.search(r'interest|hobby|hobbies', heading_lower):
            return "Interests"
        elif re.search(r'summary|profile|objective|about', heading_lower):
            return "Summary"
        elif re.search(r'achievement|accomplishment', heading_lower):
            return "Achievements"
        elif re.search(r'volunteer', heading_lower):
            return "Volunteer"
        elif re.search(r'reference', heading_lower):
            return "References"
        elif re.search(r'contact|address|phone|email', heading_lower):
            return "Contact"
        else:
            return "Other"
    
    def flatten_chunks_for_embedding(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert hierarchical chunks to flat chunks for embedding and retrieval.
        Each chunk gets metadata about its hierarchy.
        """
        flat_chunks = []
        
        for section in chunks:
            # Add the main section as a chunk
            section_chunk = {
                "text": f"{section['heading']}\n{section['content']}",
                "metadata": {
                    "type": "section",
                    "heading": section['heading'],
                    "normalized_heading": section.get('metadata', {}).get('normalized_heading', 'Other'),
                    "page": section.get('metadata', {}).get('page'),
                    "bbox": section.get('metadata', {}).get('bbox')
                }
            }
            flat_chunks.append(section_chunk)
            
            # Add each subsection as a separate chunk
            for subsection in section.get("subsections", []):
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
                        "bbox": subsection.get('metadata', {}).get('bbox')
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
        """
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


# Example usage
if __name__ == "__main__":
    chunker = ResumeChunker()
    # Example: results = chunker.process_pdf("path/to/resume.pdf")