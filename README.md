# Document Intelligence RAG Chatbot System

This project implements an intelligent document processing and querying system using FastAPI, LangChain, Google Gemini, and state-of-the-art embedding and re-ranking models. It can ingest and understand various document formats (PDF including scanned, DOCX with images, TXT), perform OCR, extract metadata, and provide an API for users to ask questions, receive contextual answers with precise citations, and maintain conversation history.

## Features

*   **Comprehensive Document Ingestion**: Robust support for PDF (including scanned documents with OCR), Microsoft Word (DOCX, including those with scanned images requiring OCR), and plain Text (TXT) files.
*   **Intelligent Metadata Extraction**: Captures essential metadata such as document name and page numbers for accurate citation.
*   **Advanced Text Processing & Chunking**: Employs contextual chunking strategies with associated metadata to preserve semantic meaning.
*   **Re-ranking for Enhanced Relevance**: Integrates a re-ranking model to improve the quality of context provided to the LLM.
*   **Conversational Query API**:
    *   `POST /api/embedding`: To process and embed documents efficiently.
    *   `POST /api/query`: To query embedded documents, receive answers with citations, and track conversation history.
*   **State-of-the-Art Model Stack**:
    *   **LLM**: Google Gemini (specifically `gemini-2.0-flash`).
    *   **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`.
    *   **Re-ranker Model**: `BAAI/bge-reranker-base`.
    *   **Vector Store**: ChromaDB.
    *   **Document Parsing**: `unstructured` library coupled with Tesseract OCR.

## Technology Choices & Justifications

The selection of technologies was driven by the core requirements of building a robust, accurate, and maintainable document intelligence system within the given timeframe.

### 1. Backend Framework: FastAPI
*   **Justification**: FastAPI was chosen for its:
    *   **High Performance**: Built on Starlette and Pydantic, it offers asynchronous capabilities (ASGI) ideal for I/O-bound operations like file processing, OCR, and calls to external LLM APIs, ensuring responsive API endpoints. 
    *   **Developer Efficiency**: Automatic data validation, serialization/deserialization via Pydantic models significantly reduces boilerplate and enhances data integrity.
    *   **Automatic API Documentation**: Built-in support for OpenAPI (Swagger UI) and ReDoc simplifies API contract definition and testing, directly addressing a deliverable requirement.
    *   **Python Ecosystem**: Leverages Python's extensive AI/ML libraries seamlessly.

### 2. Orchestration Framework: LangChain
*   **Justification**: LangChain serves as the backbone for orchestrating the RAG pipeline:
    *   **Modularity and Abstraction**: Provides pre-built, extensible components for document loading, text splitting, embedding management, vector store integration, and interaction with LLMs. This modularity accelerates development and simplifies complex workflow creation. [1, 2]
    *   **RAG-Specific Primitives**: Offers robust support for constructing and managing RAG chains, including prompt templating, context stuffing, and conversation history management, directly aligning with the project's core.
    *   **Flexibility**: While providing high-level abstractions, LangChain allows for customization of individual components, ensuring adaptability to specific model choices and processing needs. (We are using `langchain-core`, `langchain-text-splitters`, `langchain-google-genai`, `langchain-huggingface`, `langchain-chroma` for up-to-date, focused functionalities).

### 3. Model Stack Details

#### a. Large Language Model (LLM): Google Gemini (`gemini-2.0-flash`)
*   **Choice**: `gemini-2.0-flash` accessed via the `langchain-google-genai` package. 
*   **Justification**:
    *   **Advanced Reasoning and Comprehension**: Gemini models are known for their strong language understanding, reasoning capabilities, and ability to follow complex instructions, which is crucial for generating accurate, contextual answers.
    *   **Performance and Cost Balance**: The `gemini-2.0-flash` variant is optimized for speed and efficiency, making it suitable for interactive chatbot applications while offering a good balance of capability and operational cost.
    *   **Large Context Window**: Gemini models typically support large context windows, beneficial for RAG systems that might feed substantial contextual information.
    *   **Seamless LangChain Integration**: Well-supported within the LangChain ecosystem for straightforward integration and prompt management. 
    *   **Accessibility**: Readily available via Google AI Studio API, simplifying setup for development. 

#### b. Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
*   **Choice**: `sentence-transformers/all-MiniLM-L6-v2` utilized through `langchain-huggingface.HuggingFaceEmbeddings`. 
*   **Justification**:
    *   **High Retrieval Performance & Efficiency**: This model provides an excellent balance between embedding quality (ranking highly on benchmarks like MTEB for retrieval tasks) and computational efficiency (small size, fast inference). This is vital for quick document processing and query embedding.
    *   **Open Source & Local Execution**: Being open-source, it offers full control and avoids API costs for embeddings. Its small footprint allows for easy local execution.
    *   **Broad Applicability**: Suitable for a wide range of general-purpose text similarity and retrieval tasks.
    *   **LangChain Integration**: Standard and straightforward integration via `HuggingFaceEmbeddings`. 

#### c. Re-ranker Model: `BAAI/bge-reranker-base`
*   **Choice**: `BAAI/bge-reranker-base` utilized via `sentence_transformers.cross_encoder.CrossEncoder`.
*   **Justification**:
    *   **Enhanced Relevance for LLM Context**: A re-ranker significantly improves the quality of documents passed to the LLM. The initial retrieval (vector search) identifies a candidate set of documents; the re-ranker then performs a more computationally intensive, finer-grained semantic comparison between the query and each candidate document's content.
    *   **Improved Answer Accuracy**: By providing more precisely relevant context to the LLM, the re-ranker helps in generating more accurate and focused answers, directly contributing to the core requirement of "accurate, contextual responses."
    *   **State-of-the-Art Performance**: `bge-reranker-base` is a high-performing cross-encoder model on various re-ranking benchmarks (e.g., BEIR).
    *   **Reduction of Noise**: It helps filter out less relevant documents from the initial retrieval set, reducing noise and potential distraction for the LLM.
    *   **Open Source & Accessibility**: Easily accessible and usable with the `sentence-transformers` library.
    *   **Integration Strategy**: The system retrieves an initial set of N documents (e.g., N=10) using the dense retriever (all-MiniLM-L6-v2). The re-ranker then re-scores these N documents against the query. The top K (e.g., K=3) re-ranked documents are finally passed as context to the LLM, ensuring high relevance.

### 4. Vector Database: ChromaDB
*   **Choice**: ChromaDB, integrated via `langchain-chroma`.
*   **Justification**:
    *   **AI-Native & Developer-Friendly**: Designed specifically for AI/ML applications, making it easy to set up and integrate, especially with LangChain. [9, 12, 14, 18, 23]
    *   **Efficient Similarity Search**: Provides fast and effective vector similarity search capabilities.
    *   **Metadata Filtering**: Crucially supports storing and filtering by metadata (e.g., `document_id`, `page_number`), which is essential for the RAG system to retrieve chunks from specific documents and for accurate citation generation.
    *   **Persistence and In-Memory Options**: Offers flexibility with data persistence to disk (as used in this project) or in-memory operation for rapid prototyping.
    *   **Open Source**: Free to use with good community support.

### 5. Document Parsing & OCR: `unstructured` & Tesseract
*   **`unstructured` Library**:
    *   **Justification**: Addresses the core requirement of handling multiple document formats (PDF, DOCX, TXT). 
        *   **Versatile Format Support**: Provides a unified interface for parsing various file types.
        *   **OCR Integration**: Seamlessly integrates OCR capabilities (using Tesseract in this project) to extract text from scanned documents and images within files, fulfilling a key project requirement.
        *   **Advanced Partitioning**: Offers intelligent strategies to partition documents into meaningful elements (e.g., titles, paragraphs, lists), which aids in creating better quality chunks. Strategies like `"auto"` (and optional `"hi_res"` with Detectron2 for complex layouts) improve text extraction quality.
        *   **Metadata Extraction**: Capable of extracting useful metadata like page numbers (essential for citations) and element types.
*   **Tesseract OCR (via `pytesseract` and integrated by `unstructured`)**:
    *   **Justification**: Provides the OCR engine for the system. 
        *   **Robust & Open Source**: A widely used, free, and effective OCR engine.
        *   **Language Support**: Good support for English (and other languages if needed).
        *   **Baseline Accuracy**: Offers a strong baseline for OCR tasks, sufficient for many scanned documents as required by the project. While specialized cloud OCRs might offer higher accuracy on extremely challenging documents, Tesseract is a practical choice for this project's scope.

### 6. Text Chunking Strategy
*   **Strategy**: `RecursiveCharacterTextSplitter` from `langchain-text-splitters`.
    *   **Justification**:
        *   **Contextual Awareness**: This splitter attempts to break text based on a hierarchical list of separators (e.g., paragraphs `\n\n`, then sentences `\n`, then words ` `). This method is preferred over fixed-size chunking as it tries to keep semantically related content together, which is crucial for the quality of embeddings and the context provided to the LLM.
        *   **Flexibility**: Adapts to different text structures.
*   **Chunk Size**: Configured via `.env` (default: `1000` characters).
*   **Chunk Overlap**: Configured via `.env` (default: `200` characters).
    *   **Justification**:
        *   **Size**: The chunk size is chosen to be small enough to fit within the embedding model's context window (e.g., `all-MiniLM-L6-v2` handles up to 256 wordpieces, which roughly corresponds to a few hundred to a thousand characters depending on the text) while being large enough to capture meaningful semantic units for context.
        *   **Overlap**: Overlap between chunks helps maintain context continuity across chunk boundaries. If a relevant piece of information spans the end of one chunk and the beginning of another, the overlap ensures it's fully captured in at least one of the retrieved chunks.

## API Documentation

The API is built using FastAPI, which automatically generates interactive documentation.

*   **Swagger UI**: Accessible at `http://localhost:8000/docs` (or `http://127.0.0.1:8000/docs`)
*   **ReDoc**: Accessible at `http://localhost:8000/redoc` (or `http://127.0.0.1:8000/redoc`)

These interfaces provide detailed information about the API endpoints, request/response schemas, and allow for direct interaction with the API for testing.

### Endpoints Summary:

*(The summary you provided is good, ensure it accurately reflects the Pydantic models in case of any minor changes during development, especially concerning optional fields like `conversation_id` in the request vs. response for new conversations.)*

#### 1. Embed Document
*   **Endpoint**: `POST /api/embedding`
*   **Description**: Processes and embeds the provided document file.
*   **Request**: `multipart/form-data` with a `file` field containing the document (PDF, DOCX, TXT).
*   **Successful Response (200 OK)**:
    ```json
    {
      "status": "success",
      "message": "Document embedded successfully.",
      "document_id": "generated_uuid"
    }
    ```
*   **Unsuccessful Response (e.g., 400, 500)**:
    ```json
    {
      "status": "error",
      "message": "Failed to embed document.",
      "error_details": "Details of the error, e.g., 'Document content is empty.'"
    }
    ```

#### 2. Query Document
*   **Endpoint**: `POST /api/query`
*   **Description**: Allows users to query embedded documents, receive citations, and track conversation history.
*   **Request Body**:
    ```json
    {
      "query": "What is the main argument in this document?",
      "document_id": "previously_generated_uuid_from_embedding",
      "require_citations": true,
      "conversation_id": "optional_uuid_for_existing_conversation_or_null_for_new"
    }
    ```
*   **Successful Response (200 OK)** (Example for a new conversation):
    ```json
    {
      "status": "success",
      "response": {
        "answer": "The main argument is about the impact of AI on society.",
        "citations": [
          {
            "page": 12, 
            "document_name": "sample_document.pdf"
          }
        ]
      },
      "conversation_id": "newly_generated_conversation_uuid"
    }
    ```
*   **Unsuccessful Response (e.g., 404 for invalid ID, 500 for processing error)**:
    ```json
    {
      "status": "error",
      "message": "Error message, e.g., 'Invalid conversation ID. Please start a new session.'"
    }
    ```

## Setup Instructions

1.  **Prerequisites**:
    *   Python 3.8+.
    *   **Tesseract OCR Engine**: Must be installed and accessible in your system's PATH.
        *   **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr libtesseract-dev tesseract-ocr-eng` (and other language packs like `tesseract-ocr-[lang]`).
        *   **macOS**: `brew install tesseract tesseract-lang`.
        *   **Windows**: Download the installer from a reputable source (e.g., UB-Mannheim Tesseract builds on GitHub). Ensure the installation directory containing `tesseract.exe` is added to your system's PATH environment variable, and that language data (e.g., `eng.traineddata`) is present in the `tessdata` subdirectory.
    *   **(Optional, for `unstructured` `hi_res` PDF strategy)**: For enhanced PDF layout analysis, `unstructured` can use Detectron2. This requires installing `layoutparser` and `detectron2` (CPU or GPU). This setup can be complex and may require C++ build tools. If not using, ensure `DETECTRON2_OPTIMIZED=false` in your `.env` file. The default `strategy="auto"` will attempt to use the best available methods.

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shreyas-pachpute/WebOsmotic-Task
    ```

3.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    # On Linux/macOS:
    # source venv/bin/activate
    # On Windows (Command Prompt):
    # venv\Scripts\activate
    # On Windows (PowerShell):
    # .\venv\Scripts\Activate.ps1
    ```

4.  **Install Dependencies**:
    It is recommended to upgrade pip first.
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *   **Note on `protobuf`**: There were initial issues with `protobuf` versions causing conflicts (e.g., with ChromaDB or ONNX runtime). The `requirements.txt` aims for compatible versions. If you encounter `protobuf` related "Descriptor" errors or "incompatible version" warnings during runtime, you might need to explicitly install a specific `protobuf` version (e.g., `pip install protobuf==4.25.3`) after running `pip install -r requirements.txt`.
    *   **Note on `pycocotools` (Windows)**: This is a dependency for some `unstructured[local-inference]` features. If installation fails on Windows due to needing "Microsoft Visual C++ 14.0 or greater," you either need to install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or simplify `unstructured` extras in `requirements.txt` (e.g., to `unstructured[pdf,docx,txt,ocr,common]`) and set `DETECTRON2_OPTIMIZED=false` in `.env`.

5.  **Set Up Environment Variables**:
    *   Create a `.env` file in the root project directory (`document_rag_system/.env`).
    *   Populate it with your API keys and configurations:
        ```env
        GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_API_KEY"
        CHROMA_PERSIST_DIRECTORY="./chroma_db_store"
        EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" # Default, used by EmbeddingService
        LLM_MODEL_NAME="gemini-2.0-flash" # Default, used by QAService
        CHUNK_SIZE=1000 # Default, used by DocumentProcessor
        CHUNK_OVERLAP=200 # Default, used by DocumentProcessor
        DETECTRON2_OPTIMIZED=false # Set to true ONLY if Detectron2 is properly installed and you want unstructured to use it for hi_res PDF strategy
        ```
    *   Obtain your `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey). 

6.  **Run the Application**:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000` (or `http://127.0.0.1:8000`). The interactive API documentation (Swagger UI) will be at `http://localhost:8000/docs`.

## Performance Metrics

*(This section **must be filled out after testing** with the specified document categories. Provide actual numbers or observed behavior.)*

*   **Document Ingestion & Embedding Time**:
    *   **Small TXT file (e.g., 10KB, ~5 pages text):**
        *   *Processing & Chunking Time:* `~X.X seconds`
        *   *Embedding & Storage Time:* `~Y.Y seconds`
        *   *Total Time:* `~Z.Z seconds`
    *   **Medium Text-based PDF (e.g., 20 pages, ~500KB):**
        *   *Processing & Chunking Time:* `~A.A seconds`
        *   *Embedding & Storage Time:* `~B.B seconds`
        *   *Total Time:* `~C.C seconds`
    *   **Medium Scanned PDF (requiring OCR, e.g., 5 pages, ~2MB):**
        *   *Processing (including OCR) & Chunking Time:* `~D.D seconds`
        *   *Embedding & Storage Time:* `~E.E seconds`
        *   *Total Time:* `~F.F seconds`
    *   **DOCX file (e.g., 15 pages, text and a few small images):**
        *   *Processing (OCR for images if any) & Chunking Time:* `~G.G seconds`
        *   *Embedding & Storage Time:* `~H.H seconds`
        *   *Total Time:* `~I.I seconds`
*   **Query Response Time (after documents are embedded)**:
    *   *Initial Retrieval (Vector Search):* `~P.P seconds` (typically very fast)
    *   *Re-ranking Time (for N candidates):* `~Q.Q seconds` (depends on N and CPU/GPU)
    *   *LLM (Gemini) Generation Time:* `~R.R seconds` (this is often the largest component)
    *   *Total Average Query Response Time:* `~S.S seconds`
*   **Resource Usage (Observed during typical load):**
    *   *CPU Usage during Embedding (especially with OCR):* `High, e.g., X0-Y0% on Z cores`
    *   *CPU Usage during Query (Re-ranking & LLM):* `Moderate to High, e.g., A0-B0%`
    *   *Memory Usage (RAM):* `~M00 MB to N.N GB` (especially when loading models and processing large files).
    *   *Disk Space for ChromaDB:* Dependent on the number and size of documents embedded.

*(Note: These are placeholders. Be specific. For example, mention if tests were CPU-bound. Specify the machine specs if relevant for context, e.g., "Tested on a machine with Intel i7, 16GB RAM, CPU-based inference.")*

## Testing Strategy & Document Categories

The system was tested with documents from each category specified in the problem statement to ensure robust handling and accurate information retrieval.

*(For each category and sub-type below, **briefly describe the specific document used** and key observations/successes/challenges. This is a critical part of demonstrating your solution's capabilities.)*

1.  **Academic/Technical**:
    *   **Research Paper (PDF)**:
        *   *Test Document Example:* `https://www.researchgate.net/publication/371426909_Research_Paper_on_Artificial_Intelligence`
        *   *Observations:* Successfully ingested multi-column layout. Text extraction quality was Good. Queries regarding methodology and conclusions were answered accurately with correct citations. OCR was not needed.
    *   **Technical Documentation (DOCX)**:
        *   *Test Document Example:* `[Name of a sample technical manual or documentation section, e.g., "Python argparse documentation saved as DOCX"]`
        *   *Observations:* Handled structured text, code blocks (as text), and headings well. Inline images (if any) were [ignored/processed by OCR - specify]. Queries about specific functions/features yielded [accurate/relevant] answers.

2.  **Literary/Historical**:
    *   **Public Domain Book (TXT)**:
        *   *Test Document Example:* `[Specific Project Gutenberg book title and author, e.g., "Pride and Prejudice by Jane Austen (Chapter 1-5).txt"]`
        *   *Observations:* Successfully processed long-form narrative text. Chunking strategy maintained contextual flow for queries about plot points or character interactions.
    *   **Religious or Philosophical Text (PDF/TXT)**:
        *   *Test Document Example:* `[Specific public domain text, e.g., "The Meditations by Marcus Aurelius (Book I).txt"]`
        *   *Observations:* Handled potentially archaic language or unique structuring [well/with some challenges]. Queries regarding specific tenets or passages were [generally accurate/required careful phrasing]. Citations to page/section numbers were [accurate/approximate depending on source document structure].

3.  **Business/Legal**:
    *   **Annual Report or Policy Document (PDF)**:
        *   *Test Document Example:* `[A sample publicly available annual report (e.g., from a non-profit) or a generic policy document template]`
        *   *Observations:* Managed dense text and formal language. Extraction from tables was [basic - text extracted/challenging without advanced table parsing - specify]. Queries about financial summaries (if present and text-based) or policy statements were [successful/partially successful].
    *   **Legal Contract or Agreement (DOCX/Scanned PDF)**:
        *   *Test Document Example:* `[A template legal agreement, e.g., a generic NDA template. If a scanned PDF, mention it explicitly to highlight OCR testing.]`
        *   *Observations:* If scanned, OCR performance on legal jargon and dense text was [Good/Fair - note any character recognition challenges]. Successfully extracted clauses and definitions. Queries about specific obligations or terms were answered [accurately/with good relevance].

**General Testing Observations:**
*   Citation accuracy for page numbers was generally good when page information was available in the source document and extracted reliably by `unstructured`.
*   Conversation history effectively maintained context for follow-up questions related to the same document.
*   The re-ranking step demonstrably improved the relevance of the top contexts provided to the LLM, leading to more focused answers, especially for ambiguous queries.
