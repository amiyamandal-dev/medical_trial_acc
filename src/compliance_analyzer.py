# Standard library imports
import os
import hashlib
import functools
import html
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable

# Third-party imports
import lancedb
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader



class LRUCache:
    """Simple LRU cache for embeddings and search results"""
    def __init__(self, maxsize: int = 500):
        self.cache: Dict[str, Any] = {}
        self.maxsize = maxsize
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.access_order.append(key)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()


# Global caches
_search_cache = LRUCache(maxsize=200)
_document_cache = LRUCache(maxsize=50)


# Load environment variables
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


class Config:
    """Application configuration"""

    # LLM Provider
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embeddings
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Directories
    DATA_DIR = Path(__file__).parent.parent / "data"
    PROTOCOLS_DIR = DATA_DIR / "protocols"
    LANCEDB_DIR = DATA_DIR / "lancedb"

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        return bool(cls.OPENAI_API_KEY)

    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration"""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "base_url": None,
        }

    @classmethod
    def get_embedding_config(cls) -> dict:
        """Get embedding configuration"""
        return {
            "provider": "openai",
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_EMBEDDING_MODEL,
        }


class KPIConditionResponse(BaseModel):
    """Step 3.1: Conditions extracted from Protocol for a KPI"""
    kpi: str = Field(description="The KPI being analyzed")
    conditions: List[str] = Field(description="Conditions/criteria extracted from protocol")
    source_sections: List[str] = Field(description="Source sections for each condition")
    source_quotes: List[str] = Field(description="Direct quotes supporting each condition")
    protocol_context: str = Field(description="Summary of KPI context in protocol")
    measurement_criteria: str = Field(description="How the KPI should be measured")


class ConditionCheckResponse(BaseModel):
    """Step 3.2: Evidence check from Requirements"""
    condition: str = Field(description="The condition being checked")
    found_in_requirements: bool = Field(description="Whether condition was found")
    evidence_text: str = Field(description="Full relevant text from requirements")
    evidence_source: str = Field(description="Source location in requirements")
    evidence_quote: str = Field(description="Direct quote from requirements")
    coverage_level: str = Field(description="'full', 'partial', or 'none'")
    gap_description: str = Field(description="What is missing if coverage is not full")


class JudgementResponse(BaseModel):
    """Step 3.3: Final compliance judgement"""
    status: str = Field(description="'followed', 'partial', 'not_followed', or 'not_applicable'")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reasoning: str = Field(description="Detailed reasoning with source references")
    evidence_summary: str = Field(description="Summary of evidence with citations")
    gaps_identified: str = Field(description="List of gaps or missing elements")




def load_document(file_path: Path) -> str:
    """Load text from PDF or text file with caching"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check cache first
    cache_key = f"{file_path}:{file_path.stat().st_mtime}"
    cached = _document_cache.get(cache_key)
    if cached:
        return cached

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(file_path)
        content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in [".txt", ".md"]:
        content = file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Cache the result
    _document_cache.set(cache_key, content)
    return content



class VectorStore:
    """LanceDB vector store for document embeddings"""

    _TABLE_NAME = "protocol_documents"

    def __init__(self):
        Config.LANCEDB_DIR.mkdir(parents=True, exist_ok=True)

        embedding_config = Config.get_embedding_config()
        self._openai_client = OpenAI(api_key=embedding_config["api_key"])
        self._embedding_model = embedding_config["model"]

        self.db = lancedb.connect(str(Config.LANCEDB_DIR))
        self.table = None

        if self._TABLE_NAME in self.db.list_tables().tables:
            self.table = self.db.open_table(self._TABLE_NAME)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API in batches"""
        all_embeddings: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._openai_client.embeddings.create(
                input=batch,
                model=self._embedding_model,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def add_document(self, doc_id: int, content: str, metadata: Optional[dict] = None) -> None:
        """Add a document with chunking (batch optimized)"""
        chunks = self._chunk_text(content)

        # Delete existing chunks for this doc_id first (if any)
        self.delete_document(doc_id)

        # Get embeddings for all chunks
        embeddings = self._get_embeddings(chunks)

        # Prepare data for LanceDB
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append({
                "id": f"doc_{doc_id}_chunk_{i}",
                "text": chunk,
                "doc_id": doc_id,
                "chunk_index": i,
                "name": metadata.get("name", "") if metadata else "",
                "type": metadata.get("type", "") if metadata else "",
                "file_path": metadata.get("file_path", "") if metadata else "",
                "vector": embedding,
            })

        if data:
            if self.table is None:
                self.table = self.db.create_table(self._TABLE_NAME, data=data)
            else:
                self.table.add(data)

    def _chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
        """
        Split clinical trial protocol into chunks optimized for retrieval.

        Strategy for Protocol Documents:
        1. First split on major section headers (e.g., "5. STUDY DESIGN")
        2. Then on subsection headers (e.g., "5.1 Inclusion Criteria")
        3. Then on paragraph breaks
        4. Finally on sentences

        This preserves document structure and keeps related content together.
        """
        # Protocol-optimized separators (order matters - most specific first)
        separators = [
            # Major section headers: "5. SECTION NAME" or "SECTION 5:"
            "\n\n(?=\\d+\\.\\s+[A-Z])",
            "\n\n(?=SECTION\\s+\\d+)",
            # Subsection headers: "5.1 " or "5.1.1 "
            "\n\n(?=\\d+\\.\\d+\\.?\\d*\\s)",
            # Appendix/Table markers
            "\n\n(?=APPENDIX|TABLE|FIGURE)",
            # Double newlines (paragraphs)
            "\n\n",
            # Single newlines
            "\n",
            # Sentence endings
            "\\. ",
            # Fallback
            " "
        ]

        # Try section-aware splitting first
        try:
            splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len,
                is_separator_regex=True,
                keep_separator=False,  # Avoid regex flag issues
            )
            chunks = splitter.split_text(text)
        except Exception:
            # Fallback to simple splitting if regex fails
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " "],
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len,
            )
            chunks = splitter.split_text(text)

        return self._add_section_context(chunks, text)

    def _add_section_context(self, chunks: List[str], full_text: str) -> List[str]:
        """Add section header context to chunks for better retrieval"""
        import re

        # Find all section headers in the document
        section_pattern = r'^(\d+\.(?:\d+\.?)*\s+[A-Z][A-Za-z\s]+)'
        sections = []
        for match in re.finditer(section_pattern, full_text, re.MULTILINE):
            sections.append((match.start(), match.group(1).strip()))

        if not sections:
            return chunks

        enhanced_chunks = []
        text_position = 0

        for chunk in chunks:
            # Find chunk position in original text
            chunk_start = full_text.find(chunk[:100], text_position)
            if chunk_start == -1:
                chunk_start = text_position

            # Find the most recent section header before this chunk
            current_section = ""
            for pos, header in sections:
                if pos <= chunk_start:
                    current_section = header
                else:
                    break

            # Prepend section context if not already present
            if current_section and not chunk.strip().startswith(current_section[:20]):
                enhanced_chunk = f"[Section: {current_section}]\n{chunk}"
            else:
                enhanced_chunk = chunk

            enhanced_chunks.append(enhanced_chunk)
            text_position = chunk_start + len(chunk) // 2

        return enhanced_chunks

    def search(self, query: str, n_results: int = 5, doc_id: Optional[int] = None) -> List[dict]:
        """Search for similar documents with caching"""
        if self.table is None:
            return []

        # Check cache first
        cache_key = f"search:{query[:100]}:{n_results}:{doc_id}"
        cached = _search_cache.get(cache_key)
        if cached:
            return cached

        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Build search query with cosine distance
        search_query = self.table.search(query_embedding, vector_column_name="vector").metric("cosine").limit(n_results)
        if doc_id is not None:
            search_query = search_query.where(f"doc_id = {doc_id}")

        results = search_query.to_list()

        matches = []
        for row in results:
            matches.append({
                "content": row.get("text", ""),
                "metadata": {
                    "doc_id": row.get("doc_id"),
                    "chunk_index": row.get("chunk_index"),
                    "name": row.get("name", ""),
                    "type": row.get("type", ""),
                    "file_path": row.get("file_path", ""),
                },
                "distance": row.get("_distance", 0),
            })

        _search_cache.set(cache_key, matches)
        return matches

    def delete_document(self, doc_id: int) -> None:
        """Delete all chunks for a document"""
        if self.table is not None:
            try:
                self.table.delete(f"doc_id = {doc_id}")
            except Exception:
                pass  # No rows to delete


class KPIComplianceAgent:
    """Agent for 3-step KPI compliance analysis using Chain-of-Thought reasoning"""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        if not Config.validate():
            raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

        self.vector_store = vector_store
        self._current_doc_id = None
        self._response_cache = LRUCache(maxsize=100)

        llm_config = Config.get_llm_config()

        # Main LLM for analysis (lower temperature for consistency)
        self.llm = ChatOpenAI(
            api_key=llm_config["api_key"],
            model=llm_config["model"],
            temperature=0.0,
            max_tokens=2000,
            request_timeout=60,
        )

        # Faster LLM for simpler tasks (step 3.2 condition checks)
        self.llm_fast = ChatOpenAI(
            api_key=llm_config["api_key"],
            model="gpt-4o-mini",  # Faster, cheaper for condition checks
            temperature=0.0,
            max_tokens=1000,
            request_timeout=30,
        )

        self._init_3step_chains()

    def _init_3step_chains(self):
        """Initialize Chain-of-Thought chains for 3-step analysis"""

        # Step 3.1: Extract conditions from Protocol
        self.kpi_condition_parser = PydanticOutputParser(pydantic_object=KPIConditionResponse)
        step_3_1_template = """You are an expert clinical trial protocol analyst.

TASK: Extract all CONDITIONS and CRITERIA from the PROTOCOL document that relate to the given KPI.

KPI TO ANALYZE:
{kpi}

PROTOCOL DOCUMENT SECTIONS:
{protocol_context}

---
Using chain-of-thought reasoning:

Step 1 - KPI Understanding:
Understand what this KPI means in clinical trials.

Step 2 - Protocol Search (WITH SOURCE TRACKING):
Search protocol sections for ALL mentions related to this KPI.
For EACH finding, note the section number and exact quote.

Step 3 - Condition Extraction (WITH CITATIONS):
Extract specific, verifiable conditions.
For EACH condition, cite the source section and include the direct quote.

Step 4 - Context Summary:
Summarize how this KPI is addressed in the protocol.

CRITICAL: Include source_sections and source_quotes for traceability.

---
{format_instructions}"""

        self.extract_conditions_prompt = PromptTemplate(
            input_variables=["kpi", "protocol_context"],
            partial_variables={"format_instructions": self.kpi_condition_parser.get_format_instructions()},
            template=step_3_1_template
        )
        self.extract_conditions_chain = self.extract_conditions_prompt | self.llm | self.kpi_condition_parser

        # Step 3.2: Check conditions against Requirements
        self.condition_check_parser = PydanticOutputParser(pydantic_object=ConditionCheckResponse)
        step_3_2_template = """You are an expert compliance analyst.

TASK: Check if the CONDITION (from protocol) is addressed in the REQUIREMENTS document.

CONDITION TO CHECK:
{condition}

REQUIREMENTS DOCUMENT SECTIONS:
{requirements_context}

---
Using chain-of-thought reasoning:

Step 1 - Condition Understanding:
Understand what this condition requires.

Step 2 - Requirements Search (WITH SOURCE TRACKING):
Search requirements for related text. Record section identifiers and quotes.

Step 3 - Coverage Assessment:
- "full": Completely addressed
- "partial": Mentioned but incomplete
- "none": Not addressed

Step 4 - Evidence Extraction (WITH CITATIONS):
Extract evidence_text, evidence_source, and evidence_quote.

Step 5 - Gap Analysis:
Describe what is missing if coverage is not "full".

---
{format_instructions}"""

        self.check_condition_prompt = PromptTemplate(
            input_variables=["condition", "requirements_context"],
            partial_variables={"format_instructions": self.condition_check_parser.get_format_instructions()},
            template=step_3_2_template
        )
        # Use faster LLM for condition checks (simpler task)
        self.check_condition_chain = self.check_condition_prompt | self.llm_fast | self.condition_check_parser

        # Step 3.3: Judge compliance
        self.judgement_parser = PydanticOutputParser(pydantic_object=JudgementResponse)
        step_3_3_template = """You are an expert compliance judge for clinical trials.

TASK: Judge if KPI conditions from protocol are followed in requirements.

KPI:
{kpi}

CONDITIONS FROM PROTOCOL:
{conditions}

EVIDENCE FROM REQUIREMENTS:
{evidence}

---
Using chain-of-thought reasoning:

Step 1 - Evidence Review:
Review evidence for each condition with citations.

Step 2 - Compliance Assessment:
For each condition: Fully met / Partially met / Not met

Step 3 - Overall Determination:
- "followed": >=80% coverage
- "partial": 40-79% coverage
- "not_followed": <40% coverage
- "not_applicable": Not relevant

Step 4 - Confidence Score (0.0-1.0):
Based on evidence clarity and completeness.

Step 5 - Evidence Summary with citations.

Step 6 - Gap Identification:
List all gaps identified.

---
{format_instructions}"""

        self.judge_prompt = PromptTemplate(
            input_variables=["kpi", "conditions", "evidence"],
            partial_variables={"format_instructions": self.judgement_parser.get_format_instructions()},
            template=step_3_3_template
        )
        self.judge_chain = self.judge_prompt | self.llm | self.judgement_parser

    def analyze_kpi_3step(
        self,
        kpi: str,
        protocol_sections: List[Dict[str, Any]],
        requirements_sections: List[Dict[str, Any]],
        protocol_name: str,
        requirements_name: str
    ) -> Dict[str, Any]:
        """
        Perform 3-step KPI compliance analysis.

        Step 3.1: Extract conditions from Protocol
        Step 3.2: Check conditions against Requirements
        Step 3.3: Judge compliance
        """
        result = {
            "kpi": kpi,
            "requirement": kpi,  # Alias for report compatibility
            "protocol_name": protocol_name,
            "requirements_name": requirements_name,
            "step_3_1_conditions": None,
            "step_3_2_checks": [],
            "step_3_3_judgement": None,
            "compliance_status": "error",
            "confidence_score": 0.0,
            "explanation": "",
            "gaps_identified": "",
            "matched_sections": 0
        }

        try:
            # Prepare contexts
            protocol_context = self._format_sections(protocol_sections, 7, 1500)
            requirements_context = self._format_sections(requirements_sections, 7, 1500)

            # Step 3.1: Extract conditions from Protocol
            print(f"  Step 3.1: Extracting conditions for '{kpi[:50]}...'")
            conditions, source_sections, source_quotes = self._step_3_1_extract(kpi, protocol_context, result)

            # Step 3.2: Check conditions against Requirements
            print(f"  Step 3.2: Checking {len(conditions)} condition(s)...")
            evidence_list = self._step_3_2_check(
                conditions, source_sections, source_quotes, requirements_context, result
            )

            # Step 3.3: Judge compliance
            print(f"  Step 3.3: Making final judgement...")
            self._step_3_3_judge(kpi, conditions, source_sections, source_quotes, evidence_list, result)

        except Exception as e:
            result["explanation"] = f"Error during analysis: {str(e)}"
            print(f"  Analysis error: {e}")

        result["matched_sections"] = len(protocol_sections) + len(requirements_sections)
        return result

    def _format_sections(self, sections: List[Dict], max_sections: int = 5, max_chars: int = 1200) -> str:
        """Format sections for prompt context with deduplication"""
        seen_content = set()
        formatted = []

        for section in sections[:max_sections * 2]:  # Check more to find unique ones
            content = section.get('content', '')[:max_chars]
            # Simple dedup using first 100 chars as key
            content_key = content[:100].strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                formatted.append(f"Section {len(formatted)+1}:\n{content}")
                if len(formatted) >= max_sections:
                    break

        return "\n\n".join(formatted)

    def _step_3_1_extract(self, kpi: str, protocol_context: str, result: dict) -> Tuple[List, List, List]:
        """Step 3.1: Extract conditions from protocol"""
        try:
            step_result = self.extract_conditions_chain.invoke({
                "kpi": kpi,
                "protocol_context": protocol_context
            })
            result["step_3_1_conditions"] = {
                "kpi": step_result.kpi,
                "conditions": step_result.conditions,
                "source_sections": step_result.source_sections,
                "source_quotes": step_result.source_quotes,
                "protocol_context": step_result.protocol_context,
                "measurement_criteria": step_result.measurement_criteria
            }
            return step_result.conditions, step_result.source_sections, step_result.source_quotes
        except Exception as e:
            print(f"    Step 3.1 error: {e}")
            result["step_3_1_conditions"] = {
                "kpi": kpi, "conditions": [kpi],
                "source_sections": ["Unknown"], "source_quotes": ["N/A"],
                "protocol_context": "Error", "measurement_criteria": "N/A"
            }
            return [kpi], ["Unknown"], ["N/A"]

    def _step_3_2_check(
        self, conditions: List, source_sections: List, source_quotes: List,
        requirements_context: str, result: dict, concurrent: bool = True
    ) -> List[str]:
        """Step 3.2: Check conditions against requirements (with optional concurrency)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        conditions_to_check = conditions[:5]

        def check_single_condition(i: int, condition: str) -> Tuple[int, dict, str]:
            """Check a single condition"""
            cond_source = source_sections[i] if i < len(source_sections) else "Unknown"
            cond_quote = source_quotes[i] if i < len(source_quotes) else "N/A"

            try:
                step_result = self.check_condition_chain.invoke({
                    "condition": condition,
                    "requirements_context": requirements_context
                })
                check = {
                    "condition": condition,
                    "condition_source": cond_source,
                    "condition_quote": cond_quote,
                    "found_in_requirements": step_result.found_in_requirements,
                    "evidence_text": step_result.evidence_text,
                    "evidence_source": step_result.evidence_source,
                    "evidence_quote": step_result.evidence_quote,
                    "coverage_level": step_result.coverage_level,
                    "gap_description": step_result.gap_description
                }
                evidence = (
                    f"Condition (Protocol {cond_source}): {condition}\n"
                    f"Coverage: {step_result.coverage_level}\n"
                    f"Evidence Source: {step_result.evidence_source}\n"
                    f"Evidence: \"{step_result.evidence_quote[:300]}...\"\n"
                    f"Gaps: {step_result.gap_description}"
                )
                return (i, check, evidence)
            except Exception as e:
                check = {
                    "condition": condition, "condition_source": cond_source,
                    "condition_quote": cond_quote, "found_in_requirements": False,
                    "evidence_text": f"Error: {e}", "evidence_source": "N/A",
                    "evidence_quote": "N/A", "coverage_level": "none",
                    "gap_description": f"Error: {e}"
                }
                return (i, check, f"Condition: {condition}\nError: {e}")

        # Process conditions (concurrent or sequential)
        checks = [None] * len(conditions_to_check)
        evidence_list = [None] * len(conditions_to_check)

        if concurrent and len(conditions_to_check) > 1:
            # Concurrent: check all conditions in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(check_single_condition, i, cond): i
                    for i, cond in enumerate(conditions_to_check)
                }
                for future in as_completed(futures):
                    idx, check, evidence = future.result()
                    checks[idx] = check
                    evidence_list[idx] = evidence
        else:
            # Sequential: check one at a time
            for i, condition in enumerate(conditions_to_check):
                idx, check, evidence = check_single_condition(i, condition)
                checks[idx] = check
                evidence_list[idx] = evidence

        # Add results in order
        result["step_3_2_checks"] = [c for c in checks if c is not None]
        
        # Summarize step 3.2 results
        coverage_summary = {
            "full": sum(1 for c in result["step_3_2_checks"] if c.get("coverage_level") == "full"),
            "partial": sum(1 for c in result["step_3_2_checks"] if c.get("coverage_level") == "partial"),
            "none": sum(1 for c in result["step_3_2_checks"] if c.get("coverage_level") == "none")
        }
        result["step_3_2_summary"] = {
            "total_conditions_checked": len(result["step_3_2_checks"]),
            "coverage_distribution": coverage_summary,
            "summary_text": f"{coverage_summary['full']} fully covered, {coverage_summary['partial']} partially covered, {coverage_summary['none']} not covered"
        }
        
        return [e for e in evidence_list if e is not None]

    def _compute_compliance_score(self, result: dict) -> Tuple[float, str]:
        """Compute compliance score from Step 3.2 evidence checks.

        Returns (score, status) where score is 0.0-1.0 and status is the
        judgement derived from it. This ensures score and judgement are
        always consistent.
        """
        checks = result.get("step_3_2_checks", [])
        total = len(checks)
        if total == 0:
            return 0.0, "not_followed"

        full = sum(1 for c in checks if c.get("coverage_level") == "full")
        partial = sum(1 for c in checks if c.get("coverage_level") == "partial")

        score = (full + partial * 0.5) / total

        if score >= 0.8:
            status = "followed"
        elif score >= 0.4:
            status = "partial"
        else:
            status = "not_followed"

        return score, status

    def _step_3_3_judge(
        self, kpi: str, conditions: List, source_sections: List, source_quotes: List,
        evidence_list: List[str], result: dict
    ):
        """Step 3.3: Make final judgement"""
        # Summarize step 3.1 conditions
        step_3_1_summary = {
            "total_conditions": len(conditions),
            "summary_text": f"{len(conditions)} condition(s) extracted from protocol"
        }
        result["step_3_1_summary"] = step_3_1_summary

        # Compute compliance score from actual Step 3.2 evidence
        compliance_score, derived_status = self._compute_compliance_score(result)

        # Format conditions with sources
        conditions_text = "\n".join([
            f"- {cond}\n  Source: Protocol {source_sections[i] if i < len(source_sections) else 'Unknown'}"
            for i, cond in enumerate(conditions)
        ])
        evidence_text = "\n\n".join(evidence_list)

        try:
            step_result = self.judge_chain.invoke({
                "kpi": kpi,
                "conditions": conditions_text,
                "evidence": evidence_text
            })
            result["step_3_3_judgement"] = {
                "status": derived_status,
                "confidence_score": compliance_score,
                "reasoning": step_result.reasoning,
                "evidence_summary": step_result.evidence_summary,
                "gaps_identified": step_result.gaps_identified
            }
            result["compliance_status"] = derived_status
            result["confidence_score"] = compliance_score
            result["explanation"] = step_result.reasoning
            result["gaps_identified"] = step_result.gaps_identified

        except Exception as e:
            print(f"    Step 3.3 error: {e}")
            # Fallback: use computed score + collect gaps from step 3.2
            checks = result.get("step_3_2_checks", [])
            gaps = [c.get("gap_description", "") for c in checks
                    if c.get("coverage_level") in ["partial", "none"] and c.get("gap_description")]

            result["compliance_status"] = derived_status
            result["confidence_score"] = compliance_score
            result["explanation"] = f"Based on evidence: {sum(1 for c in checks if c.get('coverage_level') == 'full')}/{len(checks)} conditions fully covered"
            result["gaps_identified"] = "; ".join(gaps) if gaps else "Could not determine"
            result["step_3_3_judgement"] = {
                "status": derived_status,
                "confidence_score": compliance_score,
                "reasoning": result["explanation"],
                "evidence_summary": "Computed from condition checks",
                "gaps_identified": result["gaps_identified"]
            }

    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results"""
        total = len(results)
        followed = sum(1 for r in results if r["compliance_status"] == "followed")
        partial = sum(1 for r in results if r["compliance_status"] == "partial")
        not_followed = sum(1 for r in results if r["compliance_status"] == "not_followed")
        not_applicable = sum(1 for r in results if r["compliance_status"] == "not_applicable")
        errors = sum(1 for r in results if r["compliance_status"] == "error")

        compliance_rate = ((followed + partial * 0.5) / total * 100) if total > 0 else 0.0

        return {
            "total_requirements": total,
            "followed": followed,
            "partial": partial,
            "not_followed": not_followed,
            "not_applicable": not_applicable,
            "errors": errors,
            "compliance_rate": compliance_rate,
        }


class ComplianceAnalyzer:
    """Main interface for clinical trial compliance analysis"""

    def __init__(self, use_agent: bool = False):
        self.vector_store = VectorStore()
        self.protocol_counter = 0
        self.use_agent = use_agent
        self.agent = None
        self.content_hash_map: Dict[str, Dict[str, Any]] = {}  # hash -> protocol info

        if use_agent:
            if not Config.validate():
                raise ValueError("Agent mode requires API key in .env file.")
            self.agent = KPIComplianceAgent(vector_store=self.vector_store)

    def compute_content_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()

    def find_protocol_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Check if protocol with this content hash already exists"""
        return self.content_hash_map.get(content_hash)

    def register_protocol_hash(self, content_hash: str, protocol_info: Dict[str, Any]):
        """Register a protocol's content hash"""
        self.content_hash_map[content_hash] = protocol_info

    def load_document(self, file_path: Path) -> str:
        """Load document text"""
        return load_document(file_path)

    def analyze_kpi_compliance(
        self,
        protocol_path: Path,
        requirements_path: Path,
        kpis: List[str],
        progress_callback: Optional[callable] = None,
        concurrent: bool = False
    ) -> Dict[str, Any]:
        """
        Perform 3-step KPI compliance analysis.

        Workflow:
        - Step 3.1: KPI -> Extract conditions from Protocol
        - Step 3.2: Conditions -> Check against Requirements
        - Step 3.3: JudgeLLM -> Determine compliance

        Args:
            concurrent: If True, process KPIs in parallel (faster but uses more API calls simultaneously)
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Create with use_agent=True")

        # Load and index documents
        if progress_callback:
            progress_callback(0, 100, "Loading documents...")

        protocol_content = self.load_document(protocol_path)
        requirements_content = self.load_document(requirements_path)

        if progress_callback:
            progress_callback(5, 100, "Indexing protocol...")
        protocol_doc_id = self._add_to_store(protocol_path, protocol_content, "protocol")
        self.agent._current_doc_id = protocol_doc_id

        if progress_callback:
            progress_callback(10, 100, "Indexing requirements...")
        requirements_doc_id = self._add_to_store(requirements_path, requirements_content, "requirements")

        # Analyze KPIs (concurrent or sequential)
        if concurrent and len(kpis) > 1:
            print(f"\n[Concurrent Mode] Processing {len(kpis)} KPIs in parallel...")
            results = self._analyze_kpis_concurrent(
                kpis, protocol_doc_id, requirements_doc_id,
                protocol_path.name, requirements_path.name
            )
        else:
            results = self._analyze_kpis_sequential(
                kpis, protocol_doc_id, requirements_doc_id,
                protocol_path.name, requirements_path.name,
                progress_callback
            )

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        summary = self.agent.generate_summary(results)

        return {
            "protocol_name": protocol_path.name,
            "protocol_path": str(protocol_path),
            "requirements_name": requirements_path.name,
            "requirements_path": str(requirements_path),
            "kpis_analyzed": kpis,
            "concurrent_mode": concurrent,
            "summary": summary,
            "detailed_results": results
        }

    def _analyze_kpis_sequential(
        self,
        kpis: List[str],
        protocol_doc_id: int,
        requirements_doc_id: int,
        protocol_name: str,
        requirements_name: str,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Analyze KPIs sequentially (one at a time)"""
        results = []
        total = len(kpis)

        for i, kpi in enumerate(kpis):
            if progress_callback:
                progress = 15 + int((i / total) * 80)
                progress_callback(progress, 100, f"Analyzing KPI {i+1}/{total}")

            print(f"\n[KPI {i+1}/{total}] {kpi[:60]}...")

            protocol_sections = self.vector_store.search(kpi, n_results=10, doc_id=protocol_doc_id)
            requirements_sections = self.vector_store.search(kpi, n_results=10, doc_id=requirements_doc_id)

            result = self.agent.analyze_kpi_3step(
                kpi=kpi,
                protocol_sections=protocol_sections,
                requirements_sections=requirements_sections,
                protocol_name=protocol_name,
                requirements_name=requirements_name
            )
            results.append(result)

        return results

    def _analyze_kpis_concurrent(
        self,
        kpis: List[str],
        protocol_doc_id: int,
        requirements_doc_id: int,
        protocol_name: str,
        requirements_name: str,
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """Analyze KPIs concurrently (in parallel)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def analyze_single_kpi(kpi: str, index: int) -> Tuple[int, Dict[str, Any]]:
            """Analyze a single KPI and return with index for ordering"""
            print(f"\n[Concurrent KPI {index+1}] {kpi[:60]}...")

            protocol_sections = self.vector_store.search(kpi, n_results=10, doc_id=protocol_doc_id)
            requirements_sections = self.vector_store.search(kpi, n_results=10, doc_id=requirements_doc_id)

            result = self.agent.analyze_kpi_3step(
                kpi=kpi,
                protocol_sections=protocol_sections,
                requirements_sections=requirements_sections,
                protocol_name=protocol_name,
                requirements_name=requirements_name
            )
            return (index, result)

        # Submit all KPIs to thread pool
        results = [None] * len(kpis)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(analyze_single_kpi, kpi, i): i
                for i, kpi in enumerate(kpis)
            }

            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                    print(f"  ✓ Completed KPI {index+1}: {result.get('compliance_status', 'unknown')}")
                except Exception as e:
                    index = futures[future]
                    print(f"  ✗ Error on KPI {index+1}: {e}")
                    results[index] = {
                        "kpi": kpis[index],
                        "requirement": kpis[index],
                        "compliance_status": "error",
                        "confidence_score": 0.0,
                        "explanation": f"Error during analysis: {str(e)}",
                        "gaps_identified": "",
                        "step_3_1_conditions": None,
                        "step_3_2_checks": [],
                        "step_3_3_judgement": None,
                        "matched_sections": 0
                    }

        return results

    def _add_to_store(self, file_path: Path, content: str, doc_type: str) -> int:
        """Add document to vector store"""
        doc_id = self.protocol_counter
        self.protocol_counter += 1
        self.vector_store.add_document(
            doc_id=doc_id,
            content=content,
            metadata={"name": file_path.stem, "type": doc_type, "file_path": str(file_path)}
        )
        return doc_id

    def clear_vector_store(self):
        """Clear all data and caches"""
        if self.vector_store.table is not None:
            self.vector_store.db.drop_table(VectorStore._TABLE_NAME)
        self.vector_store = VectorStore()
        self.protocol_counter = 0
        self.content_hash_map.clear()
        # Clear global caches
        _search_cache.clear()
        _document_cache.clear()

    def clear_caches(self):
        """Clear only the caches (not the vector store)"""
        _search_cache.clear()
        _document_cache.clear()
        if self.agent:
            self.agent._response_cache.clear()


class ReportGenerator:
    """Generate formatted markdown reports"""

    @staticmethod
    def generate_markdown_report(result: Dict[str, Any], output_path: Path = None) -> str:
        """Generate markdown report from analysis results"""
        lines = [
            "# Clinical Trial Compliance Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Document Information",
            "",
            f"- **Protocol:** {result['protocol_name']}",
            f"- **Requirements:** {result['requirements_name']}",
            "",
        ]

        # Summary
        summary = result['summary']
        compliance_rate = summary['compliance_rate']
        badge = "HIGH" if compliance_rate >= 80 else "MODERATE" if compliance_rate >= 60 else "LOW"

        lines.extend([
            "## Executive Summary",
            "",
            f"### Overall Compliance: {compliance_rate:.1f}% ({badge})",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total KPIs | {summary['total_requirements']} |",
            f"| Followed | {summary['followed']} |",
            f"| Partial | {summary['partial']} |",
            f"| Not Followed | {summary['not_followed']} |",
            "",
        ])

        # HTML Executive Summary Table
        lines.append("### Summary Table")
        lines.append("")
        
        table_html = textwrap.dedent("""
            <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; font-size: 14px; margin-bottom: 20px;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                        <th style="padding: 10px; text-align: left; width: 20%;">Topic</th>
                        <th style="padding: 10px; text-align: left; width: 30%;">Protocol Summary</th>
                        <th style="padding: 10px; text-align: left; width: 30%;">Requirement Summary</th>
                        <th style="padding: 10px; text-align: center; width: 10%;">Coverage</th>
                        <th style="padding: 10px; text-align: center; width: 10%;">Judgement</th>
                    </tr>
                </thead>
                <tbody>""")

        for req in result['detailed_results']:
            kpi = req.get('kpi', req.get('requirement', 'Unknown'))
            
            # Extract Protocol Summary
            s1 = req.get('step_3_1_conditions') or {}
            protocol_summary = s1.get('protocol_context', '')
            if not protocol_summary or protocol_summary == "Error":
                 conds = s1.get('conditions', [])
                 protocol_summary = "; ".join(conds) if conds else "—"
            
            # Extract Requirement Summary
            s3 = req.get('step_3_3_judgement') or {}
            req_summary = s3.get('evidence_summary', '')
            if not req_summary:
                req_summary = "—"

            # Escape HTML content
            kpi = html.escape(kpi)
            protocol_summary = html.escape(protocol_summary)
            req_summary = html.escape(req_summary)

            # Coverage score
            conf = req.get('confidence_score', 0.0) * 100
            
            # Status styling
            status = req.get('compliance_status', 'unknown')
            status_map = {
                "followed": ("Match", "#d1fae5", "#065f46"),
                "partial": ("Partial", "#fef3c7", "#92400e"),
                "not_followed": ("No Match", "#fee2e2", "#991b1b"),
                "not_applicable": ("N/A", "#f3f4f6", "#1f2937")
            }
            status_text, bg_color, text_color = status_map.get(status, (status, "#fff", "#000"))
            
            table_html += f"""
        <tr style="border-bottom: 1px solid #e9ecef;">
            <td style="padding: 10px; vertical-align: top;"><strong>{kpi}</strong></td>
            <td style="padding: 10px; vertical-align: top;">{protocol_summary}</td>
            <td style="padding: 10px; vertical-align: top;">{req_summary}</td>
            <td style="padding: 10px; text-align: center; vertical-align: top;">{conf:.0f}%</td>
            <td style="padding: 10px; text-align: center; vertical-align: top;">
                <span style="background-color: {bg_color}; color: {text_color}; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 12px; white-space: nowrap;">{status_text}</span>
            </td>
        </tr>"""

        table_html += """
    </tbody>
</table>"""

        lines.append(table_html)
        lines.append("")

        # Alerts
        if summary['not_followed'] > 0:
            lines.append(f"**{summary['not_followed']} KPI(s) need immediate attention.**")
            lines.append("")

        # Detailed Results
        lines.extend(["---", "", "## Detailed Analysis", ""])

        for req in result['detailed_results']:
            status_icon = {"followed": "", "partial": "", "not_followed": ""}.get(
                req['compliance_status'], ""
            )
            lines.extend([
                f"### {status_icon} {req['kpi'][:80]}",
                "",
                f"**Status:** {req['compliance_status'].upper()} | **Coverage:** {req['confidence_score']*100:.0f}%",
                "",
                f"**Analysis:** {req['explanation'][:500]}",
                "",
            ])

            if req.get('gaps_identified'):
                lines.append(f"**Gaps:** {req['gaps_identified'][:300]}")
                lines.append("")

            lines.extend(["---", ""])

        # Footer
        lines.extend([
            "## Metadata",
            "",
            f"- **Analysis Method:** 3-Step KPI Compliance Analysis",
            f"- **KPIs Analyzed:** {summary['total_requirements']}",
            "",
            "*Generated by Clinical Trial Compliance Analyzer*"
        ])

        markdown = "\n".join(lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(markdown, encoding='utf-8')

        return markdown
