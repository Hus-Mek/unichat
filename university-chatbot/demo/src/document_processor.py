"""
Document Processor Module
Handles PDF and Excel document extraction and chunking
"""

import io
from typing import List
import pandas as pd
from pypdf import PdfReader

from .config import Config


class DocumentProcessor:
    """Handles document extraction and text chunking"""
    
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
    
    def extract_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Extracted text
        """
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    
    def extract_excel(self, excel_bytes: bytes) -> str:
        """
        Extract text from Excel bytes
        Preserves row structure for better chunking
        
        Args:
            excel_bytes: Excel file as bytes
            
        Returns:
            Extracted text with structure preserved
        """
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
        all_text = []
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Add sheet header
            all_text.append(f"\n=== Sheet: {sheet_name} ===")
            all_text.append(f"Columns: {', '.join(str(col) for col in df.columns)}\n")
            
            # Process each row
            for idx, row in df.iterrows():
                row_text = []
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        row_text.append(f"{col}: {value}")
                
                if row_text:
                    all_text.append(" | ".join(row_text))
                    all_text.append("")  # Empty line between rows
            
            all_text.append("\n")
        
        return "\n".join(all_text)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        lines = text.split("\n")
        buffer = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # If adding this line exceeds max length, save chunk
            if len(buffer) + len(line) > self.chunk_size and buffer:
                chunks.append(buffer.strip())
                buffer = line + "\n"
            else:
                buffer += line + "\n"
        
        # Add final chunk
        if buffer.strip():
            chunks.append(buffer.strip())
        
        return chunks
    
    def process_document(self, file_bytes: bytes, file_extension: str) -> str:
        """
        Process document based on file type
        
        Args:
            file_bytes: File content as bytes
            file_extension: File extension (.pdf, .xlsx, etc)
            
        Returns:
            Extracted text
            
        Raises:
            ValueError: If file type not supported
        """
        ext = file_extension.lower()
        
        if ext == '.pdf':
            return self.extract_pdf(file_bytes)
        elif ext in ['.xlsx', '.xls']:
            return self.extract_excel(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}")