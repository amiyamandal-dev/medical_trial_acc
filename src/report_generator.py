"""Generate markdown reports from compliance analysis results"""
from typing import Dict, Any
from datetime import datetime
from pathlib import Path


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
            compliance_badge = "🟢 **HIGH COMPLIANCE**"
        elif compliance_rate >= 60:
            compliance_badge = "🟡 **MODERATE COMPLIANCE**"
        else:
            compliance_badge = "🔴 **LOW COMPLIANCE**"

        lines.append(f"### Overall Compliance: {compliance_rate:.1f}% {compliance_badge}")
        lines.append("")

        # Summary Table
        lines.append("| Metric | Count | Percentage |")
        lines.append("|--------|-------|------------|")

        total = summary['total_requirements']
        if total > 0:
            lines.append(f"| **Total Requirements** | {total} | 100% |")
            lines.append(f"| ✅ Followed | {summary['followed']} | {(summary['followed']/total*100):.1f}% |")
            lines.append(f"| ⚠️ Partial | {summary['partial']} | {(summary['partial']/total*100):.1f}% |")
            lines.append(f"| ❌ Not Followed | {summary['not_followed']} | {(summary['not_followed']/total*100):.1f}% |")
            lines.append(f"| ➖ Not Applicable | {summary['not_applicable']} | {(summary['not_applicable']/total*100):.1f}% |")

            if summary.get('errors', 0) > 0:
                lines.append(f"| ⚠️ Errors | {summary['errors']} | {(summary['errors']/total*100):.1f}% |")

        lines.append("")

        # Key Findings
        lines.append("## Key Findings")
        lines.append("")

        if summary['not_followed'] > 0:
            lines.append(f"⚠️ **{summary['not_followed']} requirement(s) not followed** - These require immediate attention.")
            lines.append("")

        if summary['partial'] > 0:
            lines.append(f"⚠️ **{summary['partial']} requirement(s) partially met** - These may need clarification or modification.")
            lines.append("")

        if summary['followed'] == total:
            lines.append("✅ **Full compliance achieved** - All requirements are met by the protocol.")
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
            ("not_followed", "❌ Requirements Not Followed", "red"),
            ("partial", "⚠️ Partially Met Requirements", "orange"),
            ("followed", "✅ Fully Compliant Requirements", "green"),
            ("not_applicable", "➖ Non-Applicable Requirements", "gray"),
            ("error", "⚠️ Analysis Errors", "red")
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
            lines.append("✅ The protocol demonstrates strong compliance with the requirements.")
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
