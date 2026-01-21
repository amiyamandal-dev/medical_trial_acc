"""Intelligent agent for clinical trial requirements analysis using LangChain"""
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from .config import Config


# Pydantic models for structured LLM responses
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
