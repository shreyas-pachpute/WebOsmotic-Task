import os
from typing import List, Dict, Any
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import settings
from app.core.errors import DocumentProcessingError

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

    def process_document(self, file_path: str, document_name: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")

        try:
            elements = partition(filename=file_path, strategy="auto", ocr_languages="eng") 
        except Exception as e:
            raise DocumentProcessingError(f"Failed to partition document {document_name}: {str(e)}")

        chunks = []
        current_page_number = None
        doc_content = []

        for el in elements:
            page_number_in_element = el.metadata.page_number if hasattr(el.metadata, 'page_number') else None
            
            if page_number_in_element is not None:
                current_page_number = page_number_in_element
            
            text = clean(el.text, bullets=True, extra_whitespace=True, dashes=True)

            if text.strip(): 
                metadata = {
                    "document_name": document_name,
                    "page_number": current_page_number if current_page_number is not None else 1, 
                    "source_element_type": str(type(el).__name__)
                }
                doc_content.append(Document(page_content=text, metadata=metadata))

        if not doc_content:
             return [] 
        split_docs = self.text_splitter.split_documents(doc_content)
        return split_docs