"""
rag.py
Retrieval-Augmented Generation (RAG) module.

Responsibilities
-----------------
  - Load & chunk uploaded documents (PDF, TXT, MD, DOCX)
  - Embed chunks and store them in a local FAISS vector store
  - Persist the index to disk (./vectorstore/) so it survives restarts
  - Provide a similarity-search function used by the `query_knowledge_base`
    tool in tools.py

Embedding provider
-------------------
Mirrors the LLM provider priority used in graph.py — no extra API key
needed, the same key you already configured is reused:

    OPENAI_API_KEY  -> OpenAIEmbeddings (text-embedding-3-small)
    GOOGLE_API_KEY  -> GoogleGenerativeAIEmbeddings (models/embedding-001)
    MISTRAL_API_KEY -> MistralAIEmbeddings (mistral-embed)

Install dependencies:
    pip install langchain-community langchain-text-splitters faiss-cpu pypdf docx2txt
"""

import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_DIR = Path("vectorstore")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Embeddings factory — mirrors graph._build_llm() provider priority
# ---------------------------------------------------------------------------

def _build_embeddings():
    """
    Instantiate an embeddings model using whichever provider API key is
    available, in the same priority order as the chat LLM.
    """
    import os

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    elif os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    elif os.getenv("MISTRAL_API_KEY"):
        from langchain_mistralai import MistralAIEmbeddings
        return MistralAIEmbeddings(model="mistral-embed-2312")

    else:
        raise EnvironmentError(
            "No API key found for embeddings. Set OPENAI_API_KEY, "
            "GOOGLE_API_KEY, or MISTRAL_API_KEY in .env"
        )


# ---------------------------------------------------------------------------
# Vector store singleton (module-level cache)
# ---------------------------------------------------------------------------

_vectorstore: FAISS | None = None
_loaded_from_disk_attempted = False


def _load_existing_index() -> FAISS | None:
    """Try to load a previously persisted FAISS index from disk."""
    if not INDEX_DIR.exists():
        return None

    try:
        embeddings = _build_embeddings()
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"[RAG] Failed to load existing index from '{INDEX_DIR}': {e}")
        return None


def _get_vectorstore() -> FAISS | None:
    """
    Return the cached vector store, attempting a disk-load if empty.

    We only mark the disk-load as "attempted" (and stop retrying) once we
    have either successfully loaded an index OR confirmed that no index
    directory exists on disk at all. This avoids permanently giving up if
    the very first load attempt failed for a transient reason (e.g. the
    embeddings provider's API key wasn't configured yet at import time).
    """
    global _vectorstore, _loaded_from_disk_attempted

    if _vectorstore is None and not _loaded_from_disk_attempted:
        if INDEX_DIR.exists():
            loaded = _load_existing_index()
            if loaded is not None:
                _vectorstore = loaded
                _loaded_from_disk_attempted = True
            # else: leave _loaded_from_disk_attempted = False so we retry
            # on the next call (e.g. after the user fixes their API key).
        else:
            # No index on disk yet — nothing to load, stop retrying.
            _loaded_from_disk_attempted = True

    return _vectorstore


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def _get_loader(file_path: str):
    """Return the appropriate LangChain document loader for a file type."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path)

    elif suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")

    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. "
            f"Supported types: .pdf, .txt, .md, .docx"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(file_path: str, filename: str) -> int:
    """
    Load a file from disk, split it into chunks, embed, and add it to the
    vector store. The updated index is persisted to ./vectorstore/.

    Args:
        file_path: Path to the temporary/uploaded file on disk.
        filename:  Original filename (stored in chunk metadata as `source`).

    Returns:
        The number of chunks added to the index.
    """
    global _vectorstore, _loaded_from_disk_attempted

    loader = _get_loader(file_path)
    docs = loader.load()

    if not docs:
        return 0

    for doc in docs:
        doc.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        return 0

    embeddings = _build_embeddings()

    current = _get_vectorstore()
    if current is None:
        _vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        current.add_documents(chunks)
        _vectorstore = current

    INDEX_DIR.mkdir(exist_ok=True)
    _vectorstore.save_local(str(INDEX_DIR))
    _loaded_from_disk_attempted = True

    return len(chunks)


def retrieve_from_kb(query: str, k: int = 4) -> str:
    """
    Run a similarity search against the knowledge base and return formatted
    excerpts for the LLM to use as context.

    Args:
        query: The natural-language question or search query.
        k:     Number of chunks to retrieve (default 4).

    Returns:
        A formatted string of relevant excerpts, or a friendly message if
        the knowledge base is empty / nothing relevant was found.
    """
    vs = _get_vectorstore()

    if vs is None:
        return (
            "📚 The knowledge base is empty — no documents have been "
            "uploaded yet. Ask the user to upload a document first."
        )

    try:
        results = vs.similarity_search(query, k=k)
    except Exception as e:
        return f"❌ Knowledge base search failed: {e}"

    if not results:
        return f"📚 No relevant information found in the knowledge base for: '{query}'"

    lines = [f"📚 **Knowledge base results for:** {query}\n"]
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = source if page is None else f"{source} (page {page + 1})"

        content = doc.page_content.strip().replace("\n", " ")
        if len(content) > 800:
            content = content[:800] + "…"

        lines.append(f"**{i}. Source: {location}**")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


def kb_stats() -> dict:
    """
    Return basic stats about the knowledge base, for display in the UI.

    Returns:
        {"exists": bool, "chunks": int}
    """
    vs = _get_vectorstore()
    if vs is None:
        return {"exists": False, "chunks": 0}
    return {"exists": True, "chunks": vs.index.ntotal}


def clear_kb() -> None:
    """Delete the persisted vector store and clear the in-memory cache."""
    global _vectorstore, _loaded_from_disk_attempted

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    _vectorstore = None
    _loaded_from_disk_attempted = False