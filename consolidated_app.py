#!/usr/bin/env python3
"""
Clinical Trial Protocol Compliance Analysis - Consolidated Application

This single-file application provides:
1. Basic vector similarity matching between protocols and requirements
2. Intelligent LLM-based compliance analysis
3. Visual results display with charts and detailed breakdowns
4. Report generation and download functionality

Usage:
    streamlit run consolidated_app.py
"""

# =============================================================================
# EXTERNAL IMPORTS
# =============================================================================

# Standard library imports
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from pypdf import PdfReader
import streamlit as st


# =============================================================================
# GLOBAL CONFIGURATION / CONSTANTS
# =============================================================================

# Load environment variables
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


class Config:
    """Application configuration"""

    # LLM Provider Configuration
    # Set to "openai" or "deepseek"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # DeepSeek Configuration (alternative)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

    # Embedding Provider Configuration
    # Set to "openai" or "sentence-transformers"
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Sentence Transformers (alternative)
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

    # Paths
    DATA_DIR = Path(__file__).parent / "data"
    PROTOCOLS_DIR = DATA_DIR / "protocols"
    REQUIREMENTS_DIR = DATA_DIR / "requirements"

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration based on selected providers"""
        if cls.LLM_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                return False
        elif cls.LLM_PROVIDER == "deepseek":
            if not cls.DEEPSEEK_API_KEY:
                return False
        else:
            return False

        # Validate embedding provider
        if cls.EMBEDDING_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                return False

        return True

    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration based on provider"""
        if cls.LLM_PROVIDER == "openai":
            return {
                "api_key": cls.OPENAI_API_KEY,
                "model": cls.OPENAI_MODEL,
                "base_url": None,
            }
        elif cls.LLM_PROVIDER == "deepseek":
            return {
                "api_key": cls.DEEPSEEK_API_KEY,
                "model": cls.DEEPSEEK_MODEL,
                "base_url": cls.DEEPSEEK_BASE_URL,
            }
        else:
            raise ValueError(f"Unknown LLM provider: {cls.LLM_PROVIDER}")

    @classmethod
    def get_embedding_config(cls) -> dict:
        """Get embedding configuration based on provider"""
        if cls.EMBEDDING_PROVIDER == "openai":
            return {
                "provider": "openai",
                "api_key": cls.OPENAI_API_KEY,
                "model": cls.OPENAI_EMBEDDING_MODEL,
            }
        elif cls.EMBEDDING_PROVIDER == "sentence-transformers":
            return {
                "provider": "sentence-transformers",
                "model": cls.SENTENCE_TRANSFORMER_MODEL,
            }
        else:
            raise ValueError(f"Unknown embedding provider: {cls.EMBEDDING_PROVIDER}")


# =============================================================================
# CORE CLASS DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Pydantic Models for Structured LLM Responses
# -----------------------------------------------------------------------------

class ComplianceAnalysisResponse(BaseModel):
    """Structured response for requirement compliance analysis"""
    status: str = Field(
        description="Compliance status: 'followed', 'partial', 'not_followed', or 'not_applicable'"
    )
    confidence_score: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    explanation: str = Field(
        description="Brief explanation (2-3 sentences) of the assessment"
    )


class RequirementsListResponse(BaseModel):
    """Structured response for extracted requirements"""
    requirements: List[str] = Field(
        description="List of individual requirement statements"
    )


# -----------------------------------------------------------------------------
# VectorStore Class
# -----------------------------------------------------------------------------

class VectorStore:
    """Vector store using ChromaDB for document embeddings"""

    CHROMA_PATH = Path(__file__).parent / "data" / "chroma_db"
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


# -----------------------------------------------------------------------------
# ClinicalTrialAgent Class
# -----------------------------------------------------------------------------

class ClinicalTrialAgent:
    """Agent for intelligent analysis of clinical trial requirements compliance using LangChain"""

    def __init__(self):
        if not Config.validate():
            provider = Config.LLM_PROVIDER
            raise ValueError(
                f"{provider.upper()}_API_KEY not found in environment. "
                f"Please add it to your .env file. Current provider: {provider}"
            )

        # Get LLM configuration based on provider
        llm_config = Config.get_llm_config()

        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            temperature=0.0,
            max_tokens=500,
        )

        self.provider = Config.LLM_PROVIDER

        # Output parsers
        self.str_parser = StrOutputParser()
        self.compliance_parser = PydanticOutputParser(pydantic_object=ComplianceAnalysisResponse)
        self.requirements_parser = PydanticOutputParser(pydantic_object=RequirementsListResponse)

        # Initialize chains
        self._init_chains()

    def _init_chains(self):
        """Initialize LangChain chains for different tasks"""

        # Chain for extracting requirements with Pydantic validation
        self.extract_requirements_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing clinical trial documents."),
            ("user", """Extract individual requirements from this clinical trial requirements document.
Each requirement should be a clear, specific statement that can be checked against a protocol.

REQUIREMENTS DOCUMENT:
{requirements_text}

Extract only clear, specific requirements. Do not include general statements, headers, or document metadata.

{format_instructions}""")
        ])

        self.extract_requirements_chain = (
            self.extract_requirements_prompt
            | self.llm
            | self.requirements_parser
        )

        # Chain for analyzing individual requirement with Pydantic validation
        self.analyze_requirement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert clinical trial protocol analyst."),
            ("user", """Analyze whether the following requirement is met by the protocol.

REQUIREMENT:
{requirement_text}

RELEVANT PROTOCOL SECTIONS:
{context}

Analyze the requirement against the protocol sections and provide:
1. COMPLIANCE STATUS: One of: "followed", "partial", "not_followed", or "not_applicable"
   - followed: Requirement is fully met by the protocol
   - partial: Requirement is partially addressed but incomplete
   - not_followed: Requirement is not met or contradicted
   - not_applicable: Requirement does not apply to this protocol

2. CONFIDENCE SCORE: A number between 0.0 and 1.0 indicating your confidence

3. EXPLANATION: A brief explanation (2-3 sentences) of your assessment

{format_instructions}""")
        ])

        self.analyze_requirement_chain = (
            self.analyze_requirement_prompt
            | self.llm
            | self.compliance_parser
        )

    def analyze_requirement(
        self,
        requirement_text: str,
        protocol_sections: List[Dict[str, Any]],
        protocol_name: str
    ) -> Dict[str, Any]:
        """
        Analyze a single requirement against protocol sections using LangChain

        Args:
            requirement_text: The requirement to check
            protocol_sections: List of relevant protocol sections
            protocol_name: Name of the protocol being analyzed

        Returns:
            Dictionary with compliance status, score, and explanation
        """
        # Prepare context from protocol sections
        context = "\n\n".join([
            f"Section {i+1}:\n{section.get('content', '')[:1000]}"
            for i, section in enumerate(protocol_sections[:3])
        ])

        try:
            # Invoke the chain with format instructions
            result = self.analyze_requirement_chain.invoke({
                "requirement_text": requirement_text,
                "context": context,
                "format_instructions": self.compliance_parser.get_format_instructions()
            })

            # Validate status
            valid_statuses = ["followed", "partial", "not_followed", "not_applicable"]
            if result.status not in valid_statuses:
                result.status = "not_applicable"

            return {
                "requirement": requirement_text,
                "protocol_name": protocol_name,
                "compliance_status": result.status,
                "confidence_score": result.confidence_score,
                "explanation": result.explanation,
                "matched_sections": len(protocol_sections)
            }

        except Exception as e:
            return {
                "requirement": requirement_text,
                "protocol_name": protocol_name,
                "compliance_status": "error",
                "confidence_score": 0.0,
                "explanation": f"Error during analysis: {str(e)}",
                "matched_sections": 0
            }

    def extract_requirements(self, requirements_text: str) -> List[str]:
        """
        Extract individual requirements from a requirements document

        Args:
            requirements_text: Full requirements document text

        Returns:
            List of individual requirement statements
        """
        # First, try simple regex parsing for numbered requirements (fast)
        requirements = []
        lines = requirements_text.split('\n')

        for line in lines:
            line = line.strip()
            # Match patterns like "1.1 requirement text" or "2.3 requirement text"
            if line and len(line) > 15:
                # Check if line starts with a numbered pattern
                if re.match(r'^\d+\.\d+\s+', line) or re.match(r'^\d+\.\s+[A-Z]', line):
                    # Extract requirement text (after number)
                    parts = re.split(r'^\d+\.\d*\s+', line, 1)
                    if len(parts) > 1:
                        req = parts[1].strip()
                        if len(req) > 10 and not req.isupper():  # Skip headers
                            requirements.append(req)

        if requirements:
            return requirements

        # Fallback: Use LangChain LLM extraction with Pydantic
        try:
            result = self.extract_requirements_chain.invoke({
                "requirements_text": requirements_text[:5000],
                "format_instructions": self.requirements_parser.get_format_instructions()
            })

            # Filter out very short items
            requirements = [req for req in result.requirements if len(req) > 10]

            return requirements if requirements else [requirements_text[:500]]

        except Exception as e:
            # Fallback: split by pattern
            print(f"Warning: Could not extract requirements using LLM: {e}")
            requirements = []
            for line in requirements_text.split('\n'):
                if re.match(r'^\d+\.\d+\s+', line):
                    req = re.sub(r'^\d+\.\d+\s+', '', line).strip()
                    if len(req) > 10:
                        requirements.append(req)
            return requirements if requirements else [requirements_text[:500]]

    def analyze_compliance(
        self,
        requirements_text: str,
        protocol_sections: List[Dict[str, Any]],
        protocol_name: str,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze full compliance of requirements against protocol using LangChain

        Args:
            requirements_text: Full requirements document text
            protocol_sections: Relevant protocol sections from vector search
            protocol_name: Name of the protocol
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of analysis results for each requirement
        """
        # Extract individual requirements
        if progress_callback:
            progress_callback(0, 100, "Extracting requirements...")

        requirements = self.extract_requirements(requirements_text)

        if progress_callback:
            progress_callback(10, 100, f"Found {len(requirements)} requirements")

        # Analyze each requirement
        results = []
        total = len(requirements)

        for i, req in enumerate(requirements):
            if progress_callback:
                progress = 10 + int((i / total) * 80)
                progress_callback(progress, 100, f"Analyzing requirement {i+1}/{total}")

            result = self.analyze_requirement(req, protocol_sections, protocol_name)
            results.append(result)

        if progress_callback:
            progress_callback(100, 100, "Analysis complete!")

        return results

    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of compliance analysis results"""
        total = len(results)
        followed = sum(1 for r in results if r["compliance_status"] == "followed")
        partial = sum(1 for r in results if r["compliance_status"] == "partial")
        not_followed = sum(1 for r in results if r["compliance_status"] == "not_followed")
        not_applicable = sum(1 for r in results if r["compliance_status"] == "not_applicable")
        errors = sum(1 for r in results if r["compliance_status"] == "error")

        compliance_rate = 0.0
        if total > 0:
            compliance_rate = ((followed + (partial * 0.5)) / total) * 100

        return {
            "total_requirements": total,
            "followed": followed,
            "partial": partial,
            "not_followed": not_followed,
            "not_applicable": not_applicable,
            "errors": errors,
            "compliance_rate": compliance_rate,
        }


# -----------------------------------------------------------------------------
# DocumentProcessor Class
# -----------------------------------------------------------------------------

class DocumentProcessor:
    """Simple processor for matching requirements to protocol documents"""

    def __init__(self, use_agent: bool = False):
        self.vector_store = VectorStore()
        self.protocol_counter = 0
        self.use_agent = use_agent
        self.agent = None

        if use_agent:
            if not Config.validate():
                raise ValueError(
                    "Agent mode requires DEEPSEEK_API_KEY in .env file. "
                    "Run without --analyze flag for basic similarity matching."
                )
            self.agent = ClinicalTrialAgent()

    def load_document(self, file_path: Path) -> str:
        """Load text from PDF or text file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)

        elif suffix in [".txt", ".md"]:
            return file_path.read_text(encoding="utf-8")

        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md")

    def process_single_match(
        self,
        protocol_path: Path,
        requirements_path: Path
    ) -> dict:
        """
        Process a single protocol against requirements document

        Args:
            protocol_path: Path to protocol document
            requirements_path: Path to requirements document

        Returns:
            Dictionary with match results
        """
        # Load documents
        protocol_content = self.load_document(protocol_path)
        requirements_content = self.load_document(requirements_path)

        # Add protocol to vector store
        doc_id = self.protocol_counter
        self.protocol_counter += 1

        metadata = {
            "name": protocol_path.stem,
            "type": "protocol",
            "file_path": str(protocol_path)
        }

        self.vector_store.add_document(
            doc_id=doc_id,
            content=protocol_content,
            metadata=metadata
        )

        # Search for matches
        results = self.vector_store.search(
            query=requirements_content[:2000],  # Use first 2000 chars as query
            n_results=5,
            doc_id=doc_id
        )

        # Calculate average similarity score
        avg_score = 0.0
        if results:
            distances = [r.get("distance", 1.0) for r in results]
            avg_score = 1.0 - (sum(distances) / len(distances))

        return {
            "protocol_name": protocol_path.name,
            "protocol_path": str(protocol_path),
            "requirements_name": requirements_path.name,
            "requirements_path": str(requirements_path),
            "similarity_score": avg_score,
            "top_matches": results[:3]  # Top 3 matching sections
        }

    def find_best_protocol(
        self,
        protocols_dir: Path,
        requirements_path: Path
    ) -> Tuple[Optional[dict], List[dict]]:
        """
        Find the best matching protocol from a directory of protocols

        Args:
            protocols_dir: Directory containing protocol documents
            requirements_path: Path to requirements document

        Returns:
            Tuple of (best_match_result, all_results)
        """
        protocols_dir = Path(protocols_dir)

        if not protocols_dir.is_dir():
            raise ValueError(f"Not a directory: {protocols_dir}")

        # Find all protocol files
        protocol_files = []
        for ext in [".pdf", ".txt", ".md"]:
            protocol_files.extend(protocols_dir.glob(f"*{ext}"))

        if not protocol_files:
            raise ValueError(f"No protocol documents found in {protocols_dir}")

        # Load requirements
        requirements_content = self.load_document(requirements_path)

        # Process each protocol
        all_results = []
        for protocol_path in protocol_files:
            doc_id = self.protocol_counter
            self.protocol_counter += 1

            try:
                protocol_content = self.load_document(protocol_path)

                metadata = {
                    "name": protocol_path.stem,
                    "type": "protocol",
                    "file_path": str(protocol_path)
                }

                self.vector_store.add_document(
                    doc_id=doc_id,
                    content=protocol_content,
                    metadata=metadata
                )
            except Exception as e:
                print(f"Warning: Could not process {protocol_path.name}: {e}")
                continue

        # Search each protocol individually to ensure we get results for all
        protocol_scores = {}

        for i in range(len(protocol_files)):
            doc_id = i  # We assigned sequential IDs above
            file_path = str(protocol_files[i])

            # Search this specific protocol
            results = self.vector_store.search(
                query=requirements_content[:2000],
                n_results=5,
                doc_id=doc_id
            )

            if results:
                scores = [1.0 - r.get("distance", 1.0) for r in results]
                protocol_scores[file_path] = {
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "name": protocol_files[i].stem,
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "match_count": len(scores)
                }

        # Create all_results list
        for protocol_data in protocol_scores.values():
            all_results.append({
                "protocol_name": Path(protocol_data["file_path"]).name,
                "protocol_path": protocol_data["file_path"],
                "requirements_name": requirements_path.name,
                "requirements_path": str(requirements_path),
                "similarity_score": protocol_data["avg_score"],
                "max_similarity_score": protocol_data["max_score"],
                "matching_sections": protocol_data["match_count"]
            })

        # Sort by average score
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        best_match = all_results[0] if all_results else None

        return best_match, all_results

    def analyze_compliance(
        self,
        protocol_path: Path,
        requirements_path: Path,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Perform intelligent compliance analysis using LLM agent

        Args:
            protocol_path: Path to protocol document
            requirements_path: Path to requirements document
            progress_callback: Optional callback(current, total, message)

        Returns:
            Dictionary with detailed compliance analysis
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Create processor with use_agent=True")

        # Load documents
        if progress_callback:
            progress_callback(0, 100, "Loading documents...")

        protocol_content = self.load_document(protocol_path)
        requirements_content = self.load_document(requirements_path)

        # Add protocol to vector store
        doc_id = self.protocol_counter
        self.protocol_counter += 1

        metadata = {
            "name": protocol_path.stem,
            "type": "protocol",
            "file_path": str(protocol_path)
        }

        if progress_callback:
            progress_callback(5, 100, "Indexing protocol...")

        self.vector_store.add_document(
            doc_id=doc_id,
            content=protocol_content,
            metadata=metadata
        )

        # Search for relevant sections
        if progress_callback:
            progress_callback(10, 100, "Finding relevant protocol sections...")

        results = self.vector_store.search(
            query=requirements_content[:2000],
            n_results=10,
            doc_id=doc_id
        )

        # Run intelligent analysis
        analysis_results = self.agent.analyze_compliance(
            requirements_content,
            results,
            protocol_path.name,
            progress_callback=progress_callback
        )

        # Generate summary
        summary = self.agent.generate_summary(analysis_results)

        return {
            "protocol_name": protocol_path.name,
            "protocol_path": str(protocol_path),
            "requirements_name": requirements_path.name,
            "requirements_path": str(requirements_path),
            "summary": summary,
            "detailed_results": analysis_results
        }

    def analyze_compliance_folder(
        self,
        protocols_dir: Path,
        requirements_path: Path,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Find best protocol from folder and analyze compliance

        Args:
            protocols_dir: Directory containing protocol documents
            requirements_path: Path to requirements document
            progress_callback: Optional callback(current, total, message)

        Returns:
            Dictionary with compliance analysis for best matching protocol
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Create processor with use_agent=True")

        # Find best protocol first
        if progress_callback:
            progress_callback(0, 100, "Finding best matching protocol...")

        best_match, all_results = self.find_best_protocol(protocols_dir, requirements_path)

        if not best_match:
            raise ValueError("No matching protocols found")

        if progress_callback:
            progress_callback(20, 100, f"Best match: {best_match['protocol_name']}")

        # Load the best protocol
        protocol_path = Path(best_match["protocol_path"])
        requirements_content = self.load_document(requirements_path)

        # Get doc_id for the best protocol
        doc_id = None
        for i, result in enumerate(all_results):
            if result["protocol_path"] == best_match["protocol_path"]:
                doc_id = i
                break

        if doc_id is None:
            raise ValueError("Could not find protocol in results")

        # Search for relevant sections from the best protocol
        if progress_callback:
            progress_callback(25, 100, "Finding relevant sections...")

        results = self.vector_store.search(
            query=requirements_content[:2000],
            n_results=10,
            doc_id=doc_id
        )

        # Run intelligent analysis
        analysis_results = self.agent.analyze_compliance(
            requirements_content,
            results,
            best_match["protocol_name"],
            progress_callback=progress_callback
        )

        # Generate summary
        summary = self.agent.generate_summary(analysis_results)

        return {
            "protocol_name": best_match["protocol_name"],
            "protocol_path": best_match["protocol_path"],
            "requirements_name": requirements_path.name,
            "requirements_path": str(requirements_path),
            "all_protocols": all_results,
            "summary": summary,
            "detailed_results": analysis_results
        }

    def clear_vector_store(self):
        """Clear all data from vector store"""
        # Reinitialize vector store
        self.vector_store = VectorStore()
        self.protocol_counter = 0


# -----------------------------------------------------------------------------
# ReportGenerator Class
# -----------------------------------------------------------------------------

class ReportGenerator:
    """Generate formatted markdown reports for compliance analysis"""

    @staticmethod
    def generate_markdown_report(result: Dict[str, Any], output_path: Path = None) -> str:
        """
        Generate a comprehensive markdown report from analysis results

        Args:
            result: Analysis result dictionary containing summary and detailed_results
            output_path: Optional path to save the report. If None, returns string only.

        Returns:
            Markdown report as string
        """
        lines = []

        # Header
        lines.append("# Clinical Trial Compliance Analysis Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Document Information
        lines.append("## Document Information")
        lines.append("")
        lines.append(f"- **Protocol:** {result['protocol_name']}")
        lines.append(f"- **Requirements:** {result['requirements_name']}")
        lines.append("")

        # Executive Summary
        summary = result['summary']
        lines.append("## Executive Summary")
        lines.append("")

        compliance_rate = summary['compliance_rate']
        if compliance_rate >= 80:
            compliance_badge = "HIGH COMPLIANCE"
        elif compliance_rate >= 60:
            compliance_badge = "MODERATE COMPLIANCE"
        else:
            compliance_badge = "LOW COMPLIANCE"

        lines.append(f"### Overall Compliance: {compliance_rate:.1f}% {compliance_badge}")
        lines.append("")

        # Summary Table
        lines.append("| Metric | Count | Percentage |")
        lines.append("|--------|-------|------------|")

        total = summary['total_requirements']
        if total > 0:
            lines.append(f"| **Total Requirements** | {total} | 100% |")
            lines.append(f"| Followed | {summary['followed']} | {(summary['followed']/total*100):.1f}% |")
            lines.append(f"| Partial | {summary['partial']} | {(summary['partial']/total*100):.1f}% |")
            lines.append(f"| Not Followed | {summary['not_followed']} | {(summary['not_followed']/total*100):.1f}% |")
            lines.append(f"| Not Applicable | {summary['not_applicable']} | {(summary['not_applicable']/total*100):.1f}% |")

            if summary.get('errors', 0) > 0:
                lines.append(f"| Errors | {summary['errors']} | {(summary['errors']/total*100):.1f}% |")

        lines.append("")

        # Key Findings
        lines.append("## Key Findings")
        lines.append("")

        if summary['not_followed'] > 0:
            lines.append(f"**{summary['not_followed']} requirement(s) not followed** - These require immediate attention.")
            lines.append("")

        if summary['partial'] > 0:
            lines.append(f"**{summary['partial']} requirement(s) partially met** - These may need clarification or modification.")
            lines.append("")

        if summary['followed'] == total:
            lines.append("**Full compliance achieved** - All requirements are met by the protocol.")
            lines.append("")

        # Detailed Analysis
        lines.append("---")
        lines.append("")
        lines.append("## Detailed Requirements Analysis")
        lines.append("")

        # Group by status
        status_groups = {
            "not_followed": [],
            "partial": [],
            "followed": [],
            "not_applicable": [],
            "error": []
        }

        for req_result in result['detailed_results']:
            status = req_result['compliance_status']
            status_groups[status].append(req_result)

        # Display in priority order
        priority_order = [
            ("not_followed", "Requirements Not Followed", "red"),
            ("partial", "Partially Met Requirements", "orange"),
            ("followed", "Fully Compliant Requirements", "green"),
            ("not_applicable", "Non-Applicable Requirements", "gray"),
            ("error", "Analysis Errors", "red")
        ]

        for status_key, section_title, color in priority_order:
            requirements = status_groups[status_key]
            if not requirements:
                continue

            lines.append(f"### {section_title}")
            lines.append("")
            lines.append(f"*Found {len(requirements)} requirement(s) in this category*")
            lines.append("")

            for i, req_result in enumerate(requirements, 1):
                confidence = req_result['confidence_score'] * 100

                lines.append(f"#### {i}. {req_result['requirement'][:100]}{'...' if len(req_result['requirement']) > 100 else ''}")
                lines.append("")
                lines.append(f"**Full Requirement:**")
                lines.append(f"> {req_result['requirement']}")
                lines.append("")
                lines.append(f"**Confidence Score:** {confidence:.0f}%")
                lines.append("")
                lines.append(f"**Analysis:**")
                lines.append(f"{req_result['explanation']}")
                lines.append("")

                if req_result['matched_sections'] > 0:
                    lines.append(f"*Based on {req_result['matched_sections']} relevant protocol section(s)*")
                    lines.append("")

                lines.append("---")
                lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        if summary['not_followed'] > 0:
            lines.append("### Critical Actions Required")
            lines.append("")
            lines.append(f"1. **Address {summary['not_followed']} non-compliant requirement(s)**")
            lines.append("   - Review protocol sections related to these requirements")
            lines.append("   - Consider protocol amendments if necessary")
            lines.append("   - Document justification if requirements cannot be met")
            lines.append("")

        if summary['partial'] > 0:
            lines.append("### Recommended Actions")
            lines.append("")
            lines.append(f"1. **Clarify {summary['partial']} partially met requirement(s)**")
            lines.append("   - Review specific gaps identified in the analysis")
            lines.append("   - Enhance protocol documentation where needed")
            lines.append("   - Seek stakeholder approval for partial compliance")
            lines.append("")

        if compliance_rate >= 80:
            lines.append("### Overall Assessment")
            lines.append("")
            lines.append("The protocol demonstrates strong compliance with the requirements.")
            lines.append("Focus on addressing any remaining gaps to achieve full compliance.")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("## Analysis Metadata")
        lines.append("")
        lines.append(f"- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Protocol Document:** `{result['protocol_path']}`")
        lines.append(f"- **Requirements Document:** `{result['requirements_path']}`")
        lines.append(f"- **Analysis Method:** LangChain-powered LLM analysis with vector retrieval")
        lines.append(f"- **Requirements Analyzed:** {summary['total_requirements']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated by Clinical Trial Protocol Matcher - LangChain + DeepSeek*")
        lines.append("")

        # Join all lines
        markdown_content = "\n".join(lines)

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown_content, encoding='utf-8')

        return markdown_content

    @staticmethod
    def get_default_report_path(requirements_name: str) -> Path:
        """
        Generate a default report path based on requirements document name

        Args:
            requirements_name: Name of the requirements document

        Returns:
            Path object for the report file
        """
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = requirements_name.replace('.txt', '').replace('.pdf', '').replace('.md', '')
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in ('-', '_'))

        filename = f"{safe_name}_analysis_{timestamp}.md"

        return reports_dir / filename


# =============================================================================
# HELPER / UTILITY FUNCTIONS (Streamlit App)
# =============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "similarity"


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to temporary location"""
    temp_dir = Path(tempfile.gettempdir()) / "clinical_trial_app"
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def create_compliance_gauge(compliance_rate: float):
    """Create a gauge chart for compliance rate"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=compliance_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Compliance Rate", 'font': {'size': 24}},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ffcccc'},
                {'range': [40, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_status_pie_chart(summary: dict):
    """Create pie chart for compliance status distribution"""
    labels = []
    values = []
    colors = []

    if summary['followed'] > 0:
        labels.append('Followed')
        values.append(summary['followed'])
        colors.append('#28a745')

    if summary['partial'] > 0:
        labels.append('Partial')
        values.append(summary['partial'])
        colors.append('#ffc107')

    if summary['not_followed'] > 0:
        labels.append('Not Followed')
        values.append(summary['not_followed'])
        colors.append('#dc3545')

    if summary['not_applicable'] > 0:
        labels.append('Not Applicable')
        values.append(summary['not_applicable'])
        colors.append('#6c757d')

    if summary.get('errors', 0) > 0:
        labels.append('Errors')
        values.append(summary['errors'])
        colors.append('#dc3545')

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.3,
        textinfo='label+percent+value',
        textposition='auto'
    )])

    fig.update_layout(
        title="Requirements Status Distribution",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def display_similarity_results(results: dict):
    """Display results for similarity matching mode"""
    st.markdown("### Similarity Analysis Results")

    # Display protocol info
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Protocol:** {results['protocol_name']}")

    with col2:
        similarity_score = results['similarity_score'] * 100
        if similarity_score >= 70:
            st.success(f"**Similarity Score:** {similarity_score:.1f}%")
        elif similarity_score >= 40:
            st.warning(f"**Similarity Score:** {similarity_score:.1f}%")
        else:
            st.error(f"**Similarity Score:** {similarity_score:.1f}%")

    st.markdown("---")

    # Display top matching sections
    if results.get('top_matches'):
        st.markdown("### Top Matching Sections")

        for i, match in enumerate(results['top_matches'], 1):
            score = (1.0 - match.get('distance', 1.0)) * 100
            content = match.get('content', '')

            with st.expander(f"Match #{i} - Similarity: {score:.1f}%", expanded=(i == 1)):
                st.write(content)


def display_folder_results(best_match: dict, all_results: list):
    """Display results for folder matching mode"""
    st.markdown("### Best Matching Protocol")

    # Best match info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success(f"**Protocol:** {best_match['protocol_name']}")

    with col2:
        avg_score = best_match['similarity_score'] * 100
        st.metric("Average Score", f"{avg_score:.1f}%")

    with col3:
        max_score = best_match['max_similarity_score'] * 100
        st.metric("Max Section Score", f"{max_score:.1f}%")

    st.markdown("---")

    # All protocols ranking
    st.markdown("### All Protocols Ranking")

    # Create dataframe
    df = pd.DataFrame([
        {
            'Rank': i + 1,
            'Protocol': result['protocol_name'],
            'Avg Score (%)': f"{result['similarity_score'] * 100:.1f}",
            'Max Score (%)': f"{result['max_similarity_score'] * 100:.1f}",
            'Matching Sections': result['matching_sections']
        }
        for i, result in enumerate(all_results)
    ])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def display_compliance_results(results: dict):
    """Display results for intelligent compliance analysis"""
    st.markdown("### Compliance Analysis Report")

    # Document information
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Protocol:** {results['protocol_name']}")
    with col2:
        st.info(f"**Requirements:** {results['requirements_name']}")

    st.markdown("---")

    # Summary section
    summary = results['summary']

    # Compliance gauge and pie chart
    col1, col2 = st.columns(2)

    with col1:
        gauge_fig = create_compliance_gauge(summary['compliance_rate'])
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col2:
        pie_fig = create_status_pie_chart(summary)
        st.plotly_chart(pie_fig, use_container_width=True)

    # Metrics row
    st.markdown("### Summary Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Requirements", summary['total_requirements'])

    with col2:
        st.metric("Followed", summary['followed'])

    with col3:
        st.metric("Partial", summary['partial'])

    with col4:
        st.metric("Not Followed", summary['not_followed'])

    with col5:
        st.metric("N/A", summary['not_applicable'])

    st.markdown("---")

    # Key findings
    st.markdown("### Key Findings")

    if summary['not_followed'] > 0:
        st.error(f"**{summary['not_followed']} requirement(s) not followed** - These require immediate attention.")

    if summary['partial'] > 0:
        st.warning(f"**{summary['partial']} requirement(s) partially met** - These may need clarification or modification.")

    if summary['followed'] == summary['total_requirements']:
        st.success("**Full compliance achieved** - All requirements are met by the protocol.")

    st.markdown("---")

    # Detailed requirements analysis
    st.markdown("### Detailed Requirements Analysis")

    # Group requirements by status
    status_groups = {
        "not_followed": {"title": "Not Followed", "color": "red", "reqs": []},
        "partial": {"title": "Partial", "color": "orange", "reqs": []},
        "followed": {"title": "Followed", "color": "green", "reqs": []},
        "not_applicable": {"title": "Not Applicable", "color": "gray", "reqs": []},
        "error": {"title": "Error", "color": "red", "reqs": []}
    }

    for req_result in results['detailed_results']:
        status = req_result['compliance_status']
        if status in status_groups:
            status_groups[status]['reqs'].append(req_result)

    # Display by priority
    for status_key in ['not_followed', 'partial', 'followed', 'not_applicable', 'error']:
        group = status_groups[status_key]
        reqs = group['reqs']

        if not reqs:
            continue

        st.markdown(f"#### {group['title']} ({len(reqs)} requirement(s))")

        for i, req in enumerate(reqs, 1):
            confidence = req['confidence_score'] * 100

            with st.expander(
                f"{i}. {req['requirement'][:100]}{'...' if len(req['requirement']) > 100 else ''} (Confidence: {confidence:.0f}%)",
                expanded=False
            ):
                st.markdown(f"**Full Requirement:**")
                st.info(req['requirement'])

                st.markdown(f"**Confidence Score:** {confidence:.0f}%")

                st.markdown(f"**Analysis:**")
                st.write(req['explanation'])

                if req['matched_sections'] > 0:
                    st.caption(f"*Based on {req['matched_sections']} relevant protocol section(s)*")


def render_sidebar():
    """Render sidebar with configuration and file upload"""
    st.sidebar.markdown("## Configuration")

    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        options=["similarity", "intelligent"],
        format_func=lambda x: "Similarity Matching" if x == "similarity" else "Intelligent Analysis (LLM)",
        help="Similarity: Fast vector-based matching\nIntelligent: Detailed LLM-powered compliance analysis"
    )

    st.session_state.analysis_mode = analysis_mode

    # Show API key status for intelligent mode
    if analysis_mode == "intelligent":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### API Configuration")

        if Config.validate():
            st.sidebar.success(f"{Config.LLM_PROVIDER.upper()} API configured")
            st.sidebar.caption(f"Model: {Config.get_llm_config()['model']}")
        else:
            st.sidebar.error(f"{Config.LLM_PROVIDER.upper()}_API_KEY not found")
            st.sidebar.info("Please add your API key to the .env file")

    st.sidebar.markdown("---")

    # File upload section
    st.sidebar.markdown("## Upload Documents")

    # Protocol selection mode
    protocol_mode = st.sidebar.radio(
        "Protocol Mode",
        options=["single", "folder"],
        format_func=lambda x: "Single Protocol" if x == "single" else "Multiple Protocols (Folder)",
        help="Single: Analyze one protocol\nFolder: Find best match from multiple protocols"
    )

    # Protocol upload
    if protocol_mode == "single":
        protocol_file = st.sidebar.file_uploader(
            "Upload Protocol",
            type=["pdf", "txt", "md"],
            help="Upload a single protocol document"
        )
        protocol_folder = None
    else:
        protocol_file = None
        protocol_folder = st.sidebar.file_uploader(
            "Upload Protocol Documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload multiple protocol documents to find the best match"
        )

    # Requirements upload
    requirements_file = st.sidebar.file_uploader(
        "Upload Requirements",
        type=["pdf", "txt", "md"],
        help="Upload a requirements document"
    )

    st.sidebar.markdown("---")

    # Analyze button
    analyze_button = st.sidebar.button(
        "Start Analysis",
        type="primary",
        use_container_width=True,
        disabled=not requirements_file or (not protocol_file and not protocol_folder)
    )

    return {
        'analysis_mode': analysis_mode,
        'protocol_mode': protocol_mode,
        'protocol_file': protocol_file,
        'protocol_folder': protocol_folder,
        'requirements_file': requirements_file,
        'analyze_button': analyze_button
    }


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main application logic"""
    # Page configuration
    st.set_page_config(
        page_title="Clinical Trial Compliance Analyzer",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 3rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-followed {
            color: #28a745;
            font-weight: bold;
        }
        .status-partial {
            color: #ffc107;
            font-weight: bold;
        }
        .status-not-followed {
            color: #dc3545;
            font-weight: bold;
        }
        .status-na {
            color: #6c757d;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">Clinical Trial Compliance Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Match requirements to clinical trial protocols using vector similarity or intelligent LLM-based analysis</div>', unsafe_allow_html=True)

    # Sidebar
    sidebar_config = render_sidebar()

    # Main content area
    if not sidebar_config['analyze_button']:
        # Show welcome screen
        st.markdown("---")
        st.markdown("## Welcome!")
        st.markdown("""
        This application helps you analyze clinical trial protocol compliance with requirements documents.

        ### Features:

        #### Similarity Matching Mode
        - Fast vector-based document comparison
        - Find best matching protocols from a folder
        - View similarity scores and top matching sections

        #### Intelligent Analysis Mode (LLM-Powered)
        - Extracts individual requirements using AI
        - Analyzes each requirement against protocol
        - Provides compliance status (followed/partial/not followed)
        - Generates detailed explanations and confidence scores
        - Creates comprehensive markdown reports

        ### How to Use:

        1. **Configure** your analysis mode in the sidebar
        2. **Upload** your protocol and requirements documents
        3. **Click** "Start Analysis" to begin
        4. **Review** the results and download reports

        ### Supported File Types:
        - PDF (.pdf)
        - Text (.txt)
        - Markdown (.md)

        ---

        **Get Started:** Upload your documents in the sidebar!
        """)

        # Show example scenarios
        st.markdown("### Example Use Cases:")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **Single Protocol Analysis**

            Compare one protocol against requirements to get:
            - Overall compliance rate
            - Individual requirement status
            - Detailed explanations
            - Downloadable report
            """)

        with col2:
            st.info("""
            **Multi-Protocol Comparison**

            Upload multiple protocols to:
            - Find the best matching protocol
            - Compare all protocols side-by-side
            - Rank by similarity scores
            - Analyze the best match in detail
            """)

    else:
        # Run analysis
        analysis_mode = sidebar_config['analysis_mode']
        protocol_mode = sidebar_config['protocol_mode']
        use_agent = analysis_mode == "intelligent"

        # Validate intelligent mode
        if use_agent and not Config.validate():
            st.error(f"{Config.LLM_PROVIDER.upper()}_API_KEY not configured. Please add it to your .env file.")
            st.info("Switch to Similarity Matching mode or configure your API key to continue.")
            return

        try:
            # Save uploaded files
            requirements_path = save_uploaded_file(sidebar_config['requirements_file'])

            # Initialize processor
            with st.spinner("Initializing processor..."):
                processor = DocumentProcessor(use_agent=use_agent)

            # Single protocol mode
            if protocol_mode == "single" and sidebar_config['protocol_file']:
                protocol_path = save_uploaded_file(sidebar_config['protocol_file'])

                if analysis_mode == "similarity":
                    # Similarity matching
                    with st.spinner("Analyzing similarity..."):
                        results = processor.process_single_match(protocol_path, requirements_path)

                    display_similarity_results(results)

                else:
                    # Intelligent analysis
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(current, total, message):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(message)

                    results = processor.analyze_compliance(
                        protocol_path,
                        requirements_path,
                        progress_callback=progress_callback
                    )

                    progress_bar.empty()
                    status_text.empty()

                    st.success("Analysis complete!")

                    display_compliance_results(results)

                    # Generate and offer report download
                    st.markdown("---")
                    st.markdown("### Download Report")

                    report_markdown = ReportGenerator.generate_markdown_report(results)

                    st.download_button(
                        label="Download Markdown Report",
                        data=report_markdown,
                        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

            # Folder mode
            elif protocol_mode == "folder" and sidebar_config['protocol_folder']:
                # Save all protocol files to temp folder
                temp_dir = Path(tempfile.gettempdir()) / "clinical_trial_app" / "protocols"
                temp_dir.mkdir(parents=True, exist_ok=True)

                for uploaded_file in sidebar_config['protocol_folder']:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                if analysis_mode == "similarity":
                    # Folder similarity matching
                    with st.spinner("Analyzing all protocols..."):
                        best_match, all_results = processor.find_best_protocol(temp_dir, requirements_path)

                    if best_match:
                        display_folder_results(best_match, all_results)
                    else:
                        st.error("No matching protocols found.")

                else:
                    # Intelligent analysis of folder
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(current, total, message):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(message)

                    results = processor.analyze_compliance_folder(
                        temp_dir,
                        requirements_path,
                        progress_callback=progress_callback
                    )

                    progress_bar.empty()
                    status_text.empty()

                    st.success(f"Analysis complete! Best match: {results['protocol_name']}")

                    # Show all protocols ranking first
                    if results.get('all_protocols'):
                        st.markdown("### Protocol Ranking")

                        df = pd.DataFrame([
                            {
                                'Rank': i + 1,
                                'Protocol': result['protocol_name'],
                                'Avg Score (%)': f"{result['similarity_score'] * 100:.1f}",
                                'Max Score (%)': f"{result['max_similarity_score'] * 100:.1f}",
                                'Sections': result['matching_sections']
                            }
                            for i, result in enumerate(results['all_protocols'])
                        ])

                        st.dataframe(df, use_container_width=True, hide_index=True)

                        st.markdown("---")

                    # Display compliance results for best match
                    display_compliance_results(results)

                    # Generate and offer report download
                    st.markdown("---")
                    st.markdown("### Download Report")

                    report_markdown = ReportGenerator.generate_markdown_report(results)

                    st.download_button(
                        label="Download Markdown Report",
                        data=report_markdown,
                        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
