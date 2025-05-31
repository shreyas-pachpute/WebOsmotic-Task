import os
import shutil
import tempfile
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST
from app.api.models import (
    DocumentEmbedRequest, EmbedSuccessResponse, UnsuccessfulResponse,
    QueryRequest, QuerySuccessResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.qa_service import QAService
from app.utils.helpers import generate_unique_id
from app.core.errors import DocumentProcessingError, EmbeddingError, QueryError, InvalidConversationIDError, EmptyDocumentError, DocumentNotFoundError

router = APIRouter()

def get_document_processor():
    return DocumentProcessor()

def get_embedding_service():
    return EmbeddingService()

def get_qa_service(embedding_service: EmbeddingService = Depends(get_embedding_service)):
    return QAService(embedding_service=embedding_service)

@router.post("/embedding", response_model=EmbedSuccessResponse, responses={500: {"model": UnsuccessfulResponse}, 400: {"model": UnsuccessfulResponse}})
async def embed_document_route(
    file: UploadFile = File(..., description="The document file (PDF, DOCX, TXT) to embed."),
    processor: DocumentProcessor = Depends(get_document_processor),
    embed_service: EmbeddingService = Depends(get_embedding_service)
):
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"Could not save uploaded file: {str(e)}")
    finally:
        file.file.close()

    document_id = generate_unique_id()
    
    try:
        chunks = processor.process_document(file_path=file_path, document_name=file.filename)
        if not chunks:
            raise EmptyDocumentError()
            
        embed_service.embed_and_store_chunks(document_id=document_id, chunks=chunks)
        
        return EmbedSuccessResponse(document_id=document_id)
    
    except (DocumentProcessingError, EmbeddingError, EmptyDocumentError) as e:
        raise e 
    except Exception as e:
        return UnsuccessfulResponse(
            status="error",
            message="Failed to embed document due to an unexpected server error.",
            error_details=str(e)
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/query", response_model=QuerySuccessResponse, responses={500: {"model": UnsuccessfulResponse}, 404: {"model": UnsuccessfulResponse}})
async def query_document_route(
    request: QueryRequest,
    qa_service: QAService = Depends(get_qa_service)
):
    try:
        response_data, conv_id = qa_service.query_document(
            user_query=request.query,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
            require_citations=request.require_citations
        )
        return QuerySuccessResponse(response=response_data, conversation_id=conv_id)
    
    except (QueryError, InvalidConversationIDError, DocumentNotFoundError) as e:
        raise e
    except Exception as e:
        return UnsuccessfulResponse(
            status="error",
            message="Failed to process query due to an unexpected server error.",
            error_details=str(e)
        )