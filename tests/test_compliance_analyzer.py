#!/usr/bin/env python3
"""
Test cases for compliance_analyzer.py
Tests imports, utility functions, and core functionality
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    """Test that all imports work correctly"""

    def test_standard_library_imports(self):
        """Test standard library imports"""
        import os
        import re
        from datetime import datetime
        from pathlib import Path
        from typing import List, Dict, Any, Optional, Tuple
        assert True

    def test_third_party_imports(self):
        """Test third-party package imports"""
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils import embedding_functions
        from dotenv import load_dotenv
        from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pydantic import BaseModel, Field
        from pypdf import PdfReader
        assert True

    def test_module_import(self):
        """Test that the compliance_analyzer module imports without error"""
        from src.compliance_analyzer import (
            Config,
            ComplianceAnalysisResponse,
            RequirementsListResponse,
            load_document,
            calculate_similarity_score,
            create_document_metadata,
            VectorStore,
            ClinicalTrialAgent,
            ComplianceAnalyzer,
            ReportGenerator,
        )
        assert True


class TestUtilityFunctions:
    """Test utility functions"""

    def test_calculate_similarity_score_empty(self):
        """Test similarity score with empty results"""
        from src.compliance_analyzer import calculate_similarity_score

        result = calculate_similarity_score([])
        assert result == 0.0

    def test_calculate_similarity_score_single(self):
        """Test similarity score with single result"""
        from src.compliance_analyzer import calculate_similarity_score

        results = [{"distance": 0.2}]
        result = calculate_similarity_score(results)
        assert result == 0.8

    def test_calculate_similarity_score_multiple(self):
        """Test similarity score with multiple results"""
        from src.compliance_analyzer import calculate_similarity_score

        results = [
            {"distance": 0.1},
            {"distance": 0.3},
            {"distance": 0.2}
        ]
        result = calculate_similarity_score(results)
        # Average distance = (0.1 + 0.3 + 0.2) / 3 = 0.2
        # Similarity = 1.0 - 0.2 = 0.8
        assert result == 0.8

    def test_calculate_similarity_score_missing_distance(self):
        """Test similarity score with missing distance field"""
        from src.compliance_analyzer import calculate_similarity_score

        results = [{"other_field": "value"}]
        result = calculate_similarity_score(results)
        # Should default to distance=1.0, so similarity=0.0
        assert result == 0.0

    def test_create_document_metadata(self):
        """Test document metadata creation"""
        from src.compliance_analyzer import create_document_metadata

        file_path = Path("/path/to/document.pdf")
        metadata = create_document_metadata(file_path, "protocol")

        assert metadata["name"] == "document"
        assert metadata["type"] == "protocol"
        assert metadata["file_path"] == "/path/to/document.pdf"

    def test_create_document_metadata_default_type(self):
        """Test document metadata with default type"""
        from src.compliance_analyzer import create_document_metadata

        file_path = Path("/path/to/test.txt")
        metadata = create_document_metadata(file_path)

        assert metadata["type"] == "protocol"

    def test_load_document_file_not_found(self):
        """Test load_document with non-existent file"""
        from src.compliance_analyzer import load_document

        with pytest.raises(FileNotFoundError):
            load_document(Path("/non/existent/file.txt"))

    def test_load_document_unsupported_type(self, tmp_path):
        """Test load_document with unsupported file type"""
        from src.compliance_analyzer import load_document

        # Create a file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(test_file)

    def test_load_document_txt(self, tmp_path):
        """Test load_document with txt file"""
        from src.compliance_analyzer import load_document

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World", encoding="utf-8")

        content = load_document(test_file)
        assert content == "Hello World"

    def test_load_document_md(self, tmp_path):
        """Test load_document with markdown file"""
        from src.compliance_analyzer import load_document

        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\n\nContent", encoding="utf-8")

        content = load_document(test_file)
        assert "# Header" in content
        assert "Content" in content


class TestPydanticModels:
    """Test Pydantic model validation"""

    def test_compliance_analysis_response_valid(self):
        """Test valid ComplianceAnalysisResponse"""
        from src.compliance_analyzer import ComplianceAnalysisResponse

        response = ComplianceAnalysisResponse(
            status="followed",
            confidence_score=0.85,
            explanation="The requirement is fully met."
        )

        assert response.status == "followed"
        assert response.confidence_score == 0.85
        assert response.explanation == "The requirement is fully met."

    def test_compliance_analysis_response_confidence_bounds(self):
        """Test confidence score bounds validation"""
        from src.compliance_analyzer import ComplianceAnalysisResponse

        # Valid boundary values
        response_low = ComplianceAnalysisResponse(
            status="partial",
            confidence_score=0.0,
            explanation="Low confidence"
        )
        assert response_low.confidence_score == 0.0

        response_high = ComplianceAnalysisResponse(
            status="partial",
            confidence_score=1.0,
            explanation="High confidence"
        )
        assert response_high.confidence_score == 1.0

        # Invalid values should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            ComplianceAnalysisResponse(
                status="partial",
                confidence_score=1.5,  # > 1.0
                explanation="Invalid"
            )

    def test_requirements_list_response(self):
        """Test RequirementsListResponse"""
        from src.compliance_analyzer import RequirementsListResponse

        response = RequirementsListResponse(
            requirements=["Req 1", "Req 2", "Req 3"]
        )

        assert len(response.requirements) == 3
        assert "Req 1" in response.requirements


class TestConfig:
    """Test Config class"""

    def test_config_attributes_exist(self):
        """Test that config attributes exist"""
        from src.compliance_analyzer import Config

        assert hasattr(Config, "LLM_PROVIDER")
        assert hasattr(Config, "OPENAI_API_KEY")
        assert hasattr(Config, "OPENAI_MODEL")
        assert hasattr(Config, "EMBEDDING_PROVIDER")
        assert hasattr(Config, "DATA_DIR")
        assert hasattr(Config, "PROTOCOLS_DIR")
        assert hasattr(Config, "REQUIREMENTS_DIR")

    def test_config_get_llm_config_openai(self):
        """Test get_llm_config for OpenAI provider"""
        from src.compliance_analyzer import Config

        # Temporarily set provider to openai
        original_provider = Config.LLM_PROVIDER
        Config.LLM_PROVIDER = "openai"

        try:
            config = Config.get_llm_config()
            assert "api_key" in config
            assert "model" in config
            assert "base_url" in config
        finally:
            Config.LLM_PROVIDER = original_provider

    def test_config_get_llm_config_unknown_provider(self):
        """Test get_llm_config with unknown provider"""
        from src.compliance_analyzer import Config

        original_provider = Config.LLM_PROVIDER
        Config.LLM_PROVIDER = "unknown_provider"

        try:
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                Config.get_llm_config()
        finally:
            Config.LLM_PROVIDER = original_provider


class TestStatusExtraction:
    """Test status and confidence extraction helper methods"""

    def test_extract_status_followed(self):
        """Test status extraction for 'followed'"""
        from src.compliance_analyzer import ClinicalTrialAgent

        # Create a mock agent without full initialization
        agent = object.__new__(ClinicalTrialAgent)

        assert agent._extract_status_from_output("The requirement is followed completely.") == "followed"
        assert agent._extract_status_from_output("Status: FOLLOWED") == "followed"

    def test_extract_status_partial(self):
        """Test status extraction for 'partial'"""
        from src.compliance_analyzer import ClinicalTrialAgent

        agent = object.__new__(ClinicalTrialAgent)

        assert agent._extract_status_from_output("The requirement is partially met.") == "partial"
        assert agent._extract_status_from_output("Partial compliance detected") == "partial"

    def test_extract_status_not_followed(self):
        """Test status extraction for 'not_followed'"""
        from src.compliance_analyzer import ClinicalTrialAgent

        agent = object.__new__(ClinicalTrialAgent)

        assert agent._extract_status_from_output("The requirement is not followed.") == "not_followed"
        assert agent._extract_status_from_output("Status: not_followed") == "not_followed"

    def test_extract_status_not_applicable(self):
        """Test status extraction for 'not_applicable'"""
        from src.compliance_analyzer import ClinicalTrialAgent

        agent = object.__new__(ClinicalTrialAgent)

        assert agent._extract_status_from_output("This is not applicable to the protocol.") == "not_applicable"
        assert agent._extract_status_from_output("Status: not_applicable") == "not_applicable"

    def test_extract_status_default(self):
        """Test status extraction default case"""
        from src.compliance_analyzer import ClinicalTrialAgent

        agent = object.__new__(ClinicalTrialAgent)

        assert agent._extract_status_from_output("Some random text without status") == "not_applicable"

    def test_extract_confidence_from_output(self):
        """Test confidence score extraction"""
        from src.compliance_analyzer import ClinicalTrialAgent

        agent = object.__new__(ClinicalTrialAgent)

        assert agent._extract_confidence_from_output("confidence: 0.85") == 0.85
        assert agent._extract_confidence_from_output("Confidence Score: 0.9") == 0.9
        assert agent._extract_confidence_from_output("0.75 confidence") == 0.75

    def test_extract_confidence_clamping(self):
        """Test confidence score clamping"""
        from src.compliance_analyzer import ClinicalTrialAgent

        agent = object.__new__(ClinicalTrialAgent)

        # Values > 1 should be clamped to 1.0
        assert agent._extract_confidence_from_output("confidence: 1.5") == 1.0
        # Negative values won't match the regex pattern [0-9.]+, so default is returned
        # This is correct behavior since confidence scores should never be negative
        assert agent._extract_confidence_from_output("confidence: -0.5") == 0.7  # default

    def test_extract_confidence_default(self):
        """Test confidence score default"""
        from src.compliance_analyzer import ClinicalTrialAgent

        agent = object.__new__(ClinicalTrialAgent)

        # Should return default 0.7 if no confidence found
        assert agent._extract_confidence_from_output("No confidence mentioned") == 0.7


class TestReportGenerator:
    """Test ReportGenerator class"""

    def test_generate_markdown_report_basic(self):
        """Test basic markdown report generation"""
        from src.compliance_analyzer import ReportGenerator

        result = {
            "protocol_name": "test_protocol.pdf",
            "protocol_path": "/path/to/test_protocol.pdf",
            "requirements_name": "requirements.txt",
            "requirements_path": "/path/to/requirements.txt",
            "summary": {
                "total_requirements": 5,
                "followed": 3,
                "partial": 1,
                "not_followed": 1,
                "not_applicable": 0,
                "errors": 0,
                "compliance_rate": 70.0
            },
            "detailed_results": [
                {
                    "requirement": "Test requirement 1",
                    "compliance_status": "followed",
                    "confidence_score": 0.9,
                    "explanation": "Requirement is fully met.",
                    "matched_sections": 2
                },
                {
                    "requirement": "Test requirement 2",
                    "compliance_status": "not_followed",
                    "confidence_score": 0.8,
                    "explanation": "Requirement is not addressed.",
                    "matched_sections": 1
                }
            ]
        }

        report = ReportGenerator.generate_markdown_report(result)

        assert "# Clinical Trial Compliance Analysis Report" in report
        assert "test_protocol.pdf" in report
        assert "requirements.txt" in report
        assert "70.0%" in report
        assert "MODERATE COMPLIANCE" in report

    def test_get_default_report_path(self):
        """Test default report path generation"""
        from src.compliance_analyzer import ReportGenerator

        path = ReportGenerator.get_default_report_path("test_requirements.txt")

        assert path.parent.name == "reports"
        assert "test_requirements" in path.name
        assert path.suffix == ".md"


class TestVectorStoreChunking:
    """Test VectorStore text chunking"""

    def test_chunk_text_basic(self, tmp_path):
        """Test basic text chunking without full VectorStore init"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        text = "This is a test. " * 20  # ~320 chars
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
