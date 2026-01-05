"""ChromaDB Vector Store for document embeddings"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config


class VectorStore:
    """Vector store using ChromaDB for document embeddings"""

    CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
    COLLECTION_NAME = "protocol_documents"

    def __init__(self):
        self.CHROMA_PATH.mkdir(parents=True, exist_ok=True)

        # Get embedding configuration
        embedding_config = Config.get_embedding_config()

        # Initialize embedding function based on provider
        if embedding_config["provider"] == "openai":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=embedding_config["api_key"],
                model_name=embedding_config["model"]
            )
            self.embedding_provider = "openai"
        elif embedding_config["provider"] == "sentence-transformers":
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_config["model"]
            )
            self.embedding_provider = "sentence-transformers"
        else:
            raise ValueError(f"Unknown embedding provider: {embedding_config['provider']}")

        self.client = chromadb.PersistentClient(
            path=str(self.CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(self, doc_id: int, content: str, metadata: Optional[dict] = None) -> None:
        """Add a document to the vector store with chunking"""
        chunks = self._chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"doc_{doc_id}_chunk_{i}"
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }

            # Check if chunk already exists
            existing = self.collection.get(ids=[chunk_id])
            if existing["ids"]:
                self.collection.update(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[chunk_metadata]
                )
            else:
                self.collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[chunk_metadata]
                )

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    def search(self, query: str, n_results: int = 5, doc_id: Optional[int] = None) -> list[dict]:
        """Search for similar documents"""
        where_filter = {"doc_id": doc_id} if doc_id is not None else None

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        matches = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                matches.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })

        return matches

    def delete_document(self, doc_id: int) -> None:
        """Delete all chunks for a document"""
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"doc_id": doc_id}
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def get_all_doc_ids(self) -> set[int]:
        """Get all unique document IDs in the store"""
        results = self.collection.get()
        doc_ids = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if "doc_id" in meta:
                    doc_ids.add(meta["doc_id"])
        return doc_ids
