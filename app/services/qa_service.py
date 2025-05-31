import os
from typing import Dict, List, Tuple, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sentence_transformers.cross_encoder import CrossEncoder
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.core.errors import QueryError, InvalidConversationIDError, DocumentNotFoundError

conversation_histories: Dict[str, List[Dict[str, str]]] = {}

class QAService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.top_n_reranked = 3
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL_NAME,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.3
            )
        except Exception as e:
            raise QueryError(f"Failed to initialize LLM: {str(e)}")

        try:
            self.reranker = CrossEncoder(
                model_name_or_path='BAAI/bge-reranker-base',
                max_length=512,
                device='cpu'
            )
        except Exception as e:
            print(f"Warning: Failed to load BAAI/bge-reranker-base. Reranking will be skipped. Error: {str(e)}")
            self.reranker = None
        
        self.rag_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(
                content=(
                    "You are an intelligent assistant for question-answering tasks. "
                    "Use ONLY the following retrieved context from a document to answer the user's question. "
                    "If you don't know the answer from the provided context, or if the context is empty or states no relevant information, "
                    "say that you don't have enough information from the document to answer. "
                    "Do not make up information not present in the context. "
                    "Your answers should be concise and directly address the question. "
                    "When you use information from the context, you MUST cite the page number and document name "
                    "from the metadata of the relevant context snippet. "
                    "Format citations like this: (Source: [document_name], Page: [page_number]). "
                    "If multiple sources are used, list them all. "
                )
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Retrieved Context:\n<context>\n{context}\n</context>\n\nUser Question: {question}")
        ])

    def _format_docs(self, docs: List[Any]) -> str:
        if not docs:
            return "No relevant context found in the document for this question."
        
        formatted_docs = []
        for i, doc in enumerate(docs):
            metadata_str = f"(Source: {doc.metadata.get('document_name', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')})"
            formatted_docs.append(f"Context Snippet {i+1} {metadata_str}:\n{doc.page_content}")
        return "\n\n".join(formatted_docs)

    def _extract_citations_from_answer_and_context(self, answer: str, context_docs: List[Any]) -> List[Dict[str, Any]]:
        citations = []
        used_docs_info = set()
        
        for doc in context_docs:
            doc_name = doc.metadata.get('document_name', 'Unknown Document')
            page_num = doc.metadata.get('page_number', 'N/A')
            
            citation_key = (doc_name, page_num)
            if citation_key not in used_docs_info:
                citations.append({
                    "page": page_num,
                    "document_name": doc_name
                })
                used_docs_info.add(citation_key)
        return citations

    def query_document(self, user_query: str, document_id: str, conversation_id: str = None, require_citations: bool = True) -> Tuple[Dict[str, Any], str]:
        try:
            retriever = self.embedding_service.get_retriever(document_id=document_id)
        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise QueryError(f"Failed to get document retriever for document ID {document_id}: {str(e)}")

        initial_docs = retriever.invoke(user_query)
        context_docs = initial_docs

        if self.reranker and initial_docs:
            try:
                print(f"Reranking {len(initial_docs)} documents for query: '{user_query[:50]}...'")
                sentence_pairs = [(user_query, doc.page_content) for doc in initial_docs]
                
                if not sentence_pairs:
                     print("No sentence pairs to rerank.")
                else:
                    scores = self.reranker.predict(sentence_pairs, show_progress_bar=False)
                    scored_docs = list(zip(scores, initial_docs))
                    scored_docs.sort(key=lambda x: x[0], reverse=True)
                    context_docs = [doc for score, doc in scored_docs[:self.top_n_reranked]]
                    print(f"Selected {len(context_docs)} documents after reranking.")

            except Exception as e:
                print(f"Warning: Reranking failed. Falling back to initial retriever results limited to {self.top_n_reranked}. Error: {str(e)}")
                context_docs = initial_docs[:self.top_n_reranked]
        elif initial_docs: 
            context_docs = initial_docs[:self.top_n_reranked] 

        current_chat_history = []
        if conversation_id:
            if conversation_id not in conversation_histories:
                raise InvalidConversationIDError()
            history_tuples = conversation_histories.get(conversation_id, [])
            for entry in history_tuples:
                if entry["role"] == "user":
                    current_chat_history.append(HumanMessage(content=entry["content"]))
                elif entry["role"] == "assistant":
                    current_chat_history.append(AIMessage(content=entry["content"]))
        
        formatted_context = self._format_docs(context_docs)
        
        messages = self.rag_prompt_template.format_messages(
            context=formatted_context,
            question=user_query,
            chat_history=current_chat_history
        )
        
        try:
            llm_response = self.llm.invoke(messages)
            answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        except Exception as e:
            raise QueryError(f"Error during LLM invocation: {str(e)}")

        citations_list = []
        if require_citations and context_docs:
            citations_list = self._extract_citations_from_answer_and_context(answer, context_docs)

        response_data = {
            "answer": answer,
            "citations": citations_list
        }

        new_conv_id = conversation_id
        if not new_conv_id:
            new_conv_id = f"conv_{os.urandom(4).hex()}" 
        
        if new_conv_id not in conversation_histories:
            conversation_histories[new_conv_id] = []
        
        conversation_histories[new_conv_id].append({"role": "user", "content": user_query})
        conversation_histories[new_conv_id].append({"role": "assistant", "content": answer})

        return response_data, new_conv_id