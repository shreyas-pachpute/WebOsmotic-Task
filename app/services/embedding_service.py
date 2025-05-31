from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  
from app.core.config import settings
from app.core.errors import EmbeddingError, DocumentNotFoundError

class EmbeddingService:
    def __init__(self):
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True} 
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}")

        try:
            self.vector_store = Chroma(
                persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embedding_model
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize ChromaDB: {str(e)}")

    def embed_and_store_chunks(self, document_id: str, chunks: List[Document]):
        if not chunks:
            raise EmbeddingError("No chunks provided to embed.")
        try:
            for chunk in chunks:
                chunk.metadata["document_id"] = document_id

            self.vector_store.add_documents(documents=chunks, ids=[f"{document_id}_{i}" for i in range(len(chunks))])
        except Exception as e:
            raise EmbeddingError(f"Failed to embed and store chunks for document {document_id}: {str(e)}")

    def get_retriever(self, document_id: str, k_results: int = 5):
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    'k': k_results,
                    'filter': {'document_id': document_id}
                }
            )
            return retriever 
        except Exception as e: 
             raise DocumentNotFoundError(f"Could not create retriever or find document ID {document_id}. Ensure it's embedded. Original error: {e}")