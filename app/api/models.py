from pydantic import BaseModel, Field
from typing import List, Optional, Any

class DocumentEmbedRequest(BaseModel):
    document: str = Field(..., description="Path to the document file to be embedded.")

class EmbedSuccessResponse(BaseModel):
    status: str = "success"
    message: str = "Document embedded successfully."
    document_id: str

class ErrorDetail(BaseModel):
    loc: Optional[List[str]] = None
    msg: str
    type: str

class UnsuccessfulResponse(BaseModel):
    status: str = "error"
    message: str
    error_details: Optional[Any] = None # Can be string or structured like ErrorDetail

class Citation(BaseModel):
    page: Optional[Any] = Field(None, description="Page number of the citation. Can be int or N/A")
    document_name: str

class QueryResponseData(BaseModel):
    answer: str
    citations: List[Citation]

class QuerySuccessResponse(BaseModel):
    status: str = "success"
    response: QueryResponseData
    conversation_id: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    document_id: str
    require_citations: bool = True
    conversation_id: Optional[str] = None