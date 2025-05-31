from fastapi import HTTPException, status

class DocumentProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

class EmbeddingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

class QueryError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

class InvalidConversationIDError(HTTPException):
    def __init__(self, detail: str = "Invalid conversation ID. Please start a new session."):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class DocumentNotFoundError(HTTPException):
    def __init__(self, detail: str = "Document ID not found."):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class EmptyDocumentError(HTTPException):
    def __init__(self, detail: str = "Document content is empty or could not be processed."):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)