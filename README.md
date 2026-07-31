# Multi-Document RAG Chatbot

A high-performance, production-grade Retrieval-Augmented Generation (RAG) system for querying multiple PDF, DOCX, and TXT files. Designed with dual-column layouts, citations, progress meters, local caching, and fallback providers.

---

## 1. Setup Instructions

1.  **System Requirements**:
    *   Python 3.11.x is recommended for optimal library stability.
    *   Local Ollama installation (if running local LLM).

2.  **Clone and Navigate**:
    ```powershell
    cd "c:/Users/username/Projects/Loan RAG chatbot"
    ```

3.  **Create and Activate Virtual Environment**:
    ```powershell
    py -3.11 -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

4.  **Install Pinned Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

5.  **Configure Credentials**:
    Copy `.env.example` to `.env` and fill in API keys:
    ```powershell
    Copy-Item .env.example .env
    ```

6.  **Download the Embedding Model (one-time, requires internet)**:
    See [Section 2: First-Time Setup — Embedding Model Cache](#2-first-time-setup-embedding-model-cache) below. This step is required before the app will run, even if you already have the model cached elsewhere on your machine — the app expects it in a specific project-local folder.

7.  **Pull Local Model**:
    Verify Ollama is running and fetch Llama3:
    ```powershell
    ollama pull llama3
    ```

8.  **Launch Dashboard**:
    ```powershell
    streamlit run app.py
    ```

---

## 2. First-Time Setup: Embedding Model Cache

This project uses a local embedding model (`BAAI/bge-large-en-v1.5`) that must be downloaded once per machine before running the app. This is a one-time step after cloning the repo, separate from `pip install`.

### Why this is needed
The app runs with Hugging Face's offline mode enabled (`HF_HUB_OFFLINE=1`) so it never makes network calls at runtime. That means the model files must already exist locally in the exact folder the app expects — cloning the repo does **not** include the model weights (they're git-ignored due to size, ~1.3GB).

### Setup steps

1. Make sure you've completed steps 1–5 above (venv created, dependencies installed).

2. Run this **one-time download command** (requires internet access), from the project root:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5', cache_folder='./models')"
   ```
   This downloads the model into `./models`, matching the exact cache path the app reads from at runtime. The `cache_folder='./models'` argument is required — a plain download without it will not be found by the app.

3. Verify the download succeeded:
   ```powershell
   # Windows PowerShell
   Get-ChildItem .\models -Recurse -Filter "config.json"
   ```
   ```bash
   # macOS/Linux
   find ./models -name "config.json"
   ```
   You should see `config.json` under `models/models--BAAI--bge-large-en-v1.5/snapshots/.../`.

4. Continue to step 7 (Pull Local Model) in Section 1.

### Troubleshooting
If you see an error like `couldn't connect to 'https://huggingface.co'` when starting the app, it almost always means step 2 was skipped, run without the correct `cache_folder` argument, or run in a different directory than the project root. Re-run step 2 exactly as written above from the project root.

---

## 3. Folder Structure

```text
├── app.py                     # Streamlit Main Coordinator & Layout Router
├── requirements.txt           # Production Library Dependencies
├── requirements-lock.txt      # Fully Pinned Dependency Lockfile
├── .env.example               # Environment Configuration Template
├── config/
│   ├── settings.py            # Settings Parser & Environment Variables
│   └── __init__.py
├── ingestion/
│   ├── pdf_loader.py          # PyMuPDF & pdfplumber PDF Parser
│   ├── cleaner.py             # Text Normalizer & Hyphen Rejoiner
│   ├── metadata_builder.py    # Regex-based Section Headings Builder
│   └── __init__.py
├── chunking/
│   ├── metadata_manager.py    # Deterministic Doc Hash & ID Manager
│   ├── semantic_chunker.py    # Cosine Similarity-based Semantic Splitter
│   └── __init__.py
├── embeddings/
│   ├── embedder.py            # Singleton BGE-Large-v1.5 Embedder
│   └── __init__.py
├── vectorstore/
│   ├── chroma_manager.py      # ChromaDB Persistent Client Manager
│   └── __init__.py
├── retrievers/
│   ├── dense_retriever.py     # Dense Semantic Similarity Searcher
│   ├── bm25_retriever.py      # Sparse BM25 Keyword Searcher
│   ├── hybrid_retriever.py    # RRF (Reciprocal Rank Fusion) Ensemble Searcher
│   ├── reranker.py            # MS-Marco Cross-Encoder Reranker
│   └── __init__.py
├── llms/
│   ├── llama_provider.py      # Llama3 Local Ollama Router
│   ├── gemini_provider.py     # Google Gemini API Provider
│   ├── groq_provider.py       # Groq API Fallback Provider
│   ├── openai_provider.py     # OpenAI API Fallback Provider
│   ├── llm_router.py          # Priority-based Fallback LLM Router
│   ├── reasoning_engine.py    # Multi-Document Context Synthesis Engine
│   └── __init__.py
├── guardrails/
│   ├── input_guard.py         # Input safety/relevance check filter
│   ├── output_guard.py        # Output synthesis verification filter
│   └── __init__.py
├── utils/
│   ├── logger.py              # Central Logger Utility
│   ├── resource_manager.py    # Cached @st.cache_resource Singleton getters
│   └── __init__.py
├── ui/
│   ├── components/
│   │   ├── metrics.py         # Dashboard Statistics metrics card block
│   │   ├── sidebar.py         # Sidebar panel, Health Indicators, Reset popover
│   │   └── source_panel.py    # Right Panel cited chunk text expanders
│   └── pages/
│       ├── chat_assistant.py  # Central conversational view
│       ├── upload.py          # Document Drag and Drop upload panel
│       ├── document_viewer.py # Page/Chunk inspector and searcher
│       ├── ask_explore.py     # Advanced retrieval playground
│       └── settings.py        # Settings configuration inspector
└── models/                    # Offline local model weights cache
```

---

## 4. Migration Guide (Python 3.14 to 3.11)

To migrate from the experimental Python 3.14 environment to the stable production Python 3.11 build:
1.  **Deactivate active environments** (`deactivate`).
2.  Install Python 3.11.x on your host system.
3.  Recreate the virtual environment explicitly directing to Python 3.11 installation binary:
    ```powershell
    & "C:\Path\To\Python311\python.exe" -m venv venv
    ```
4.  Re-run `pip install -r requirements-lock.txt`.
5.  This avoids `_UnionGenericAlias` warnings and restores full compatibility for Pydantic V1/V2 namespaces.

---

## 5. Performance & Optimization Report

### Key Tuning Operations
*   **Offline Mode Enforcement**: Setting `TRANSFORMERS_OFFLINE=1` when hub caches exist limits startup network roundtrips, reducing loading delays from 30 seconds to **1 second**.
*   **Bulk Document Insertion**: Changed iterative adds in Chroma to a single `collection.add(...)` vector array insert, dropping chunk storage latencies.
*   **Session Caching**: Singletons (`ChromaManager`, `BGEEmbedder`, `CrossEncoderReranker`) are registered in Streamlit `@st.cache_resource` memory to guarantee instantiation runs once.

