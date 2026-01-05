"""Simple document processor for clinical trial protocol matching"""
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from pypdf import PdfReader
from .vector_store import VectorStore
from .agent import ClinicalTrialAgent
from .config import Config


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
