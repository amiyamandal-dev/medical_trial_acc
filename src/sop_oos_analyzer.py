"""
SOP vs OOS Compliance Analyzer

Extraction and comparison of pharmaceutical SOP documents against
OOS (Out-of-Specification) investigation reports.

Two extraction modes:
  - Deterministic: regex + PDF text parsing (pypdf) — fast, no API cost.
  - LLM-enhanced: sends raw PDF text to OpenAI for structured extraction,
    then merges with regex results for maximum accuracy.
"""

import json as _json
import os
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_pdf_text(pdf_path: str | Path) -> str:
    """Extract full text from a PDF, page by page."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _re_first(pattern: str, text: str, flags: int = 0) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _nl_field(text: str, field: str) -> str:
    """Extract a value from newline-separated table field/value pairs.

    PDF tables often render as:
        Field Name\\nValue\\n
    """
    m = re.search(re.escape(field) + r"\s*\n\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    # Fallback: colon-separated
    m = re.search(re.escape(field) + r":\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_bullets(text: str, start_pattern: str, stop_pattern: str) -> list[str]:
    """Extract bullet-pointed items between two section markers.

    Handles \\x7f, \\u2022, -, and plain lines as bullet chars.
    """
    section = re.search(
        start_pattern + r"\s*\n(.*?)" + stop_pattern,
        text, re.DOTALL | re.IGNORECASE,
    )
    if not section:
        return []
    items: list[str] = []
    for line in section.group(1).strip().splitlines():
        line = line.strip().lstrip("\x7f\u2022•-").strip()
        if line and len(line) > 3:
            items.append(line)
    return items


def _parse_range(criteria: str) -> dict[str, Any]:
    """Parse acceptance criteria string into structured bounds.

    Handles:
      '95.0% - 105.0% of labeled amount'  -> range
      'NMT 2.0%'                           -> nmt
      'NLT 80% (Q) in 30 minutes ...'     -> nlt
      'NMT 15 minutes'                     -> nmt (time)
      '4.0 - 8.0 kg/cm2'                  -> range
      'Average weight ± 5.0%'             -> pct_deviation
      'NMT 10 ppm'                         -> nmt
      'TAMC: NMT 103 CFU/g ...'           -> nmt (microbial, 10^3)
    """
    result: dict[str, Any] = {
        "lower_bound": None,
        "upper_bound": None,
        "unit": None,
        "criteria_type": None,
    }

    # Range: "95.0% - 105.0%" or "4.0 - 8.0 kg/cm2" or "15-20 kN"
    m = re.search(r"([\d.]+)\s*%?\s*[-–]\s*([\d.]+)\s*(%|kg/cm2|kN|minutes?)?", criteria)
    if m:
        result["lower_bound"] = float(m.group(1))
        result["upper_bound"] = float(m.group(2))
        result["unit"] = m.group(3) or "%"
        result["criteria_type"] = "range"
        return result

    # NMT
    m = re.search(r"NMT\s+([\d.]+)\s*(%|ppm|minutes?|°C|CFU/g|w/w)?", criteria, re.IGNORECASE)
    if m:
        result["upper_bound"] = float(m.group(1))
        result["unit"] = m.group(2) or ""
        result["criteria_type"] = "nmt"
        return result

    # NLT
    m = re.search(r"NLT\s+([\d.]+)\s*(%|ppm|minutes?)?", criteria, re.IGNORECASE)
    if m:
        result["lower_bound"] = float(m.group(1))
        result["unit"] = m.group(2) or ""
        result["criteria_type"] = "nlt"
        return result

    # ± deviation
    m = re.search(r"[±]\s*([\d.]+)%", criteria)
    if m:
        result["upper_bound"] = float(m.group(1))
        result["unit"] = "%"
        result["criteria_type"] = "pct_deviation"
        return result

    # Microbial: NMT 10^3 style (superscript renders as "103" in PDF)
    m = re.search(r"NMT\s+10(\d)\s*(CFU/g)?", criteria, re.IGNORECASE)
    if m:
        result["upper_bound"] = 10 ** int(m.group(1))
        result["unit"] = m.group(2) or "CFU/g"
        result["criteria_type"] = "nmt"
        return result

    result["criteria_type"] = "qualitative"
    return result


def _section_text(text: str, start: str, stop: str) -> str:
    """Extract text between two section heading patterns."""
    m = re.search(start + r"(.*?)" + stop, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# SOP Extraction
# ---------------------------------------------------------------------------

# The SOP CQA table (Section 6.0) renders as newline-separated columns:
#   Parameter\nAcceptance Criteria (may span lines)\nReference\nTest Method\n
_CQA_ROW_NAMES = [
    "Appearance",
    "Identification",
    "Assay (Paracetamol)",
    "Dissolution",
    "Hardness",
    "Friability",
    "Weight Variation",
    "Disintegration",
    "Related Substance J",
    "Microbial Limits",
]


def _parse_sop_cqa_table(text: str) -> list[dict[str, Any]]:
    """Parse the product specifications table from Section 6.0.

    The table in the PDF renders as newline-separated columns:
        Appearance
        White to off-white, round, biconvex tablets with
        break line on one side. Free from visible defects.
        BP/USP
        Visual
        Identification
        Positive for Paracetamol by HPLC retention time
        and UV spectrum
        BP/USP
        HPLC
        ...

    Strategy: find each parameter name, then capture everything until the
    next parameter name (or end of section). The last two tokens before the
    next param are Reference and Test Method; everything in between is the
    acceptance criteria.
    """
    # Isolate Section 6.0 content
    section = _section_text(text, r"6\.0\s+PRODUCT SPECIFICATIONS.*?\n", r"(?:Page \d|7\.0\s)")
    if not section:
        return []

    # Remove table headers
    section = re.sub(r"Parameter\s*\n\s*Acceptance Criteria\s*\n\s*Reference\s*\n\s*Test Method\s*\n?", "", section)

    lines = [l.strip() for l in section.splitlines() if l.strip()]

    # Find indices of each parameter row start (exact match only to avoid
    # false hits like "Hardness Tester" or "Disintegration Apparatus")
    param_indices: list[tuple[int, str]] = []
    seen: set[str] = set()
    for i, line in enumerate(lines):
        for pname in _CQA_ROW_NAMES:
            if pname in seen:
                continue
            if line == pname:
                param_indices.append((i, pname))
                seen.add(pname)
                break

    specs: list[dict[str, Any]] = []
    for idx, (start_i, pname) in enumerate(param_indices):
        # Determine end of this parameter's block
        end_i = param_indices[idx + 1][0] if idx + 1 < len(param_indices) else len(lines)
        block = lines[start_i + 1 : end_i]

        if len(block) < 2:
            # Not enough lines for criteria + ref + method
            criteria_text = " ".join(block) if block else ""
            ref = ""
            method = ""
        else:
            # Last line is test method, second-to-last is reference
            method = block[-1]
            ref = block[-2]
            criteria_lines = block[:-2]
            criteria_text = " ".join(criteria_lines)

            # Validate ref looks like a reference (BP, USP, Internal, etc.)
            if not re.match(r"^(BP|USP|Internal|BP/USP|USP/BP)", ref, re.IGNORECASE):
                # ref is probably part of criteria, method might be ref
                if re.match(r"^(BP|USP|Internal|BP/USP|USP/BP)", method, re.IGNORECASE):
                    criteria_text = " ".join(criteria_lines + [ref])
                    ref = method
                    method = ""
                else:
                    criteria_text = " ".join(criteria_lines + [ref])
                    ref = ""

        bounds = _parse_range(criteria_text)
        specs.append({
            "parameter": pname,
            "acceptance_criteria": criteria_text,
            "reference": ref,
            "test_method": method,
            **bounds,
        })

    return specs


def _parse_newline_table(text: str, row_names: list[str], num_cols: int) -> list[dict[str, list[str]]]:
    """Generic parser for newline-separated tables with known row names.

    Returns list of dicts with 'name' and 'values' keys.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Sort row_names longest-first so "Column Temperature" matches before "Column"
    sorted_names = sorted(row_names, key=len, reverse=True)
    indices: list[tuple[int, str]] = []
    matched_lines: set[int] = set()
    for i, line in enumerate(lines):
        if i in matched_lines:
            continue
        for rn in sorted_names:
            if line == rn or line.startswith(rn):
                indices.append((i, rn))
                matched_lines.add(i)
                break
    # Re-sort by line index
    indices.sort(key=lambda x: x[0])

    results = []
    for idx, (start_i, rn) in enumerate(indices):
        end_i = indices[idx + 1][0] if idx + 1 < len(indices) else len(lines)
        values = lines[start_i + 1 : end_i]
        results.append({"name": rn, "values": values})
    return results


def extract_sop_data(pdf_path: str | Path) -> dict[str, Any]:
    """Extract structured data from an SOP PDF."""
    text = _read_pdf_text(pdf_path)

    # ── Metadata ──
    doc_id = _re_first(r"Document\s+ID:\s*([\w-]+)", text) or ""
    effective_date = _re_first(r"Effective\s+Date:\s*([\w-]+)", text) or ""
    revision = _re_first(r"Revision:\s*(\d+)", text) or ""
    title = _re_first(r"STANDARD OPERATING PROCEDURE\s*\n\s*(.+)", text) or ""
    department = _nl_field(text, "Department")
    applicable_to = _nl_field(text, "Applicable To")
    supersedes = _nl_field(text, "Supersedes")
    review_frequency = _nl_field(text, "Review Frequency")

    # ── Objective (Section 1.0) ──
    objective = _section_text(text, r"1\.0\s+OBJECTIVE\s*\n", r"2\.0\s+SCOPE")
    objective = re.sub(r"\s+", " ", objective).strip()

    # ── Scope (Section 2.0) ──
    scope_section = _section_text(text, r"2\.0\s+SCOPE\s*\n", r"3\.0\s+RESPONSIBILITIES")
    scope_intro = ""
    scope_items: list[str] = []
    for line in scope_section.splitlines():
        line = line.strip().lstrip("\x7f\u2022•").strip()
        if not line:
            continue
        if scope_items or line.startswith("All ") or line.startswith("Quality") or line.startswith("Documentation"):
            scope_items.append(line)
        else:
            scope_intro += " " + line
    scope_intro = scope_intro.strip()

    # ── Responsibilities (Section 3.0) ──
    resp_section = _section_text(text, r"3\.0\s+RESPONSIBILITIES\s*\n", r"4\.0\s+DEFINITIONS")
    # Remove table header
    resp_section = re.sub(r"Role\s*\n\s*Responsibilities\s*\n?", "", resp_section)
    responsibilities: list[dict[str, str]] = []
    role_names = ["QC Analyst", "QC Supervisor", "QC Manager", "QA Manager"]
    resp_rows = _parse_newline_table(resp_section, role_names, 1)
    for row in resp_rows:
        responsibilities.append({
            "role": row["name"],
            "responsibilities": " ".join(row["values"]),
        })

    # ── Definitions & Abbreviations (Section 4.0) ──
    def_section = _section_text(text, r"4\.0\s+DEFINITIONS AND ABBREVIATIONS\s*\n", r"5\.0\s+ENVIRONMENTAL")
    def_section = re.sub(r"Term\s*\n\s*Definition\s*\n?", "", def_section)
    # Remove page footers that may appear in the section
    def_section = re.sub(r"Page \d+ of \d+\n.*?CONFIDENTIAL.*?\n?", "", def_section)
    definitions: list[dict[str, str]] = []
    abbrev_names = [
        "API", "BP", "CAPA", "CGMP", "HPLC", "LIMS",
        "NLT", "NMT", "OOS", "RH", "SOP", "USP",
    ]
    def_rows = _parse_newline_table(def_section, abbrev_names, 1)
    for row in def_rows:
        definitions.append({
            "term": row["name"],
            "definition": " ".join(row["values"]),
        })

    # ── Environmental Controls (Section 5.0) ──
    env_section = _section_text(text, r"5\.0\s+ENVIRONMENTAL CONTROLS\s*\n", r"6\.0\s+PRODUCT")
    env_section = re.sub(r"Parameter\s*\n\s*Specification\s*\n\s*Monitoring Frequency\s*\n?", "", env_section)
    environmental: list[dict[str, Any]] = []
    env_names = ["Temperature", "Relative Humidity", "Particulate Matter"]
    env_rows = _parse_newline_table(env_section, env_names, 2)
    for row in env_rows:
        vals = row["values"]
        spec = vals[0] if vals else ""
        freq = vals[1] if len(vals) > 1 else ""
        entry: dict[str, Any] = {
            "parameter": row["name"],
            "specification": spec,
            "monitoring_frequency": freq,
        }
        nums = re.findall(r"[\d.]+", spec)
        if len(nums) >= 2:
            entry["lower_bound"] = float(nums[0])
            entry["upper_bound"] = float(nums[1])
        environmental.append(entry)

    # ── Product Specifications (Section 6.0) ──
    specifications = _parse_sop_cqa_table(text)

    # ── Reagents and Solvents (Section 7.1) ──
    reagents_section = _section_text(text, r"7\.1\s+Reagents and Solvents\s*\n", r"7\.2\s+Equipment")
    reagents_section = re.sub(r"Item\s*\n\s*Grade\s*\n\s*Storage\s*\n?", "", reagents_section)
    reagent_names = [
        "Methanol", "Acetonitrile", "Potassium Dihydrogen Phosphate",
        "Sodium Hydroxide", "Paracetamol Reference Standard", "Water",
    ]
    reagents: list[dict[str, str]] = []
    rea_rows = _parse_newline_table(reagents_section, reagent_names, 2)
    for row in rea_rows:
        vals = row["values"]
        reagents.append({
            "item": row["name"],
            "grade": vals[0] if vals else "",
            "storage": vals[1] if len(vals) > 1 else "",
        })

    # ── Equipment (Section 7.2) ──
    equip_section = _section_text(text, r"7\.2\s+Equipment\s*\n", r"8\.0\s+ANALYTICAL")
    equip_section = re.sub(r"Equipment\s*\n\s*Specification\s*\n\s*Calibration Frequency\s*\n?", "", equip_section)
    equip_names = [
        "HPLC System", "Analytical Balance", "Dissolution Apparatus",
        "UV-VIS Spectrophotometer", "Hardness Tester", "Friabilator",
    ]
    equipment: list[dict[str, str]] = []
    eq_rows = _parse_newline_table(equip_section, equip_names, 2)
    for row in eq_rows:
        vals = row["values"]
        # Values may span multiple lines (e.g. HPLC spec wraps)
        if len(vals) >= 2:
            spec_text = " ".join(vals[:-1])
            cal_text = vals[-1]
        elif vals:
            spec_text = vals[0]
            cal_text = ""
        else:
            spec_text = cal_text = ""
        equipment.append({
            "equipment": row["name"],
            "specification": spec_text,
            "calibration_frequency": cal_text,
        })

    # ── Chromatographic Conditions (Section 8.1) ──
    chrom_section = _section_text(text, r"8\.1\s+Chromatographic Conditions\s*\n", r"8\.2\s+System Suitability")
    chrom_section = re.sub(r"Parameter\s*\n\s*Specification\s*\n?", "", chrom_section)
    chrom_params = [
        "Column", "Mobile Phase", "Flow Rate", "Injection Volume",
        "Detection Wavelength", "Column Temperature", "Run Time",
        "Retention Time (Expected)",
    ]
    # "Column Temperature" must come before "Column" to avoid partial match
    chrom_params.sort(key=len, reverse=True)
    chromatographic_conditions: list[dict[str, str]] = []
    chrom_rows = _parse_newline_table(chrom_section, chrom_params, 1)
    for row in chrom_rows:
        chromatographic_conditions.append({
            "parameter": row["name"],
            "specification": " ".join(row["values"]),
        })

    # ── System Suitability Requirements (Section 8.2) ──
    # These span pages 3-4. The header is on page 3, values on page 4.
    ss_section = _section_text(text, r"8\.2\s+System Suitability Requirements\s*\n", r"8\.3\s+Sample Preparation")
    ss_section = re.sub(r"Parameter\s*\n\s*Acceptance Criteria\s*\n?", "", ss_section)
    # Remove page footers
    ss_section = re.sub(r"Page \d+ of \d+\n.*?CONFIDENTIAL.*?\n?", "", ss_section)
    ss_params = ["Tailing Factor", "Theoretical Plates", "%RSD of Peak Area", "Resolution"]

    system_suitability: list[dict[str, Any]] = []
    ss_rows = _parse_newline_table(ss_section, ss_params, 1)
    for row in ss_rows:
        raw = " ".join(row["values"]).strip()
        # Clean up any extraneous content (e.g. "(n=5)" or "(from nearest peak)")
        bounds = _parse_range(raw)
        system_suitability.append({
            "parameter": row["name"],
            "acceptance_criteria": raw,
            **bounds,
        })

    # ── Sample Preparation (Section 8.3) ──
    sample_prep_steps = _extract_numbered_steps(
        text, r"8\.3\s+Sample Preparation\s*\n", r"8\.4\s+Standard Preparation"
    )

    # ── Standard Preparation (Section 8.4) ──
    standard_prep_steps = _extract_numbered_steps(
        text, r"8\.4\s+Standard Preparation\s*\n", r"8\.5\s+Calculation"
    )

    # ── Calculation Formula (Section 8.5) ──
    calc_section = _section_text(text, r"8\.5\s+Calculation\s*\n", r"(?:Page \d|9\.0\s)")
    calc_formula = ""
    calc_variables: list[dict[str, str]] = []
    for line in calc_section.splitlines():
        line = line.strip()
        if "Label Claim" in line or "Asample" in line and "=" in line and "×" in line:
            calc_formula = line
        m = re.match(r"^(\w+)\s*=\s*(.+)", line)
        if m and m.group(1) not in ("Where",):
            calc_variables.append({"symbol": m.group(1), "description": m.group(2).strip()})

    # ── Data Integrity (Section 9.0) ──
    data_integrity_items = _extract_bullets(
        text, r"9\.0\s+DATA INTEGRITY AND DOCUMENTATION", r"10\.0\s+OUT-OF-SPECIFICATION"
    )
    # Also capture the intro sentence
    di_intro = _section_text(text, r"9\.0\s+DATA INTEGRITY AND DOCUMENTATION\s*\n", r"\x7f")
    di_intro = re.sub(r"\s+", " ", di_intro).strip()

    # ── OOS Handling Procedure (Section 10.0) ──
    oos_steps = _extract_numbered_steps(
        text, r"10\.0\s+OUT-OF-SPECIFICATION.*?\n", r"11\.0\s+REFERENCES"
    )
    # Also capture the intro text
    oos_intro = _section_text(
        text, r"10\.0\s+OUT-OF-SPECIFICATION.*?\n",
        r"(?:1\.\s+STOP|$)"
    )
    oos_intro = re.sub(r"\s+", " ", oos_intro).strip()

    # ── References (Section 11.0) ──
    references = _extract_bullets(text, r"11\.0\s+REFERENCES", r"12\.0\s+REVISION")

    # ── Revision History (Section 12.0) ──
    rev_section = _section_text(text, r"12\.0\s+REVISION HISTORY\s*\n", r"(?:Page \d|DOCUMENT APPROVAL)")
    rev_section = re.sub(r"Rev\.\s*\n\s*Date\s*\n\s*Description of Change\s*\n\s*Author\s*\n?", "", rev_section)
    revision_history: list[dict[str, str]] = []
    # Revision rows: Rev#\nDate\nDescription\nAuthor
    rev_matches = re.findall(r"(\d+)\s*\n\s*(\d{2}-\w{3}-\d{4})\s*\n\s*(.+?)\s*\n\s*(.+?)(?:\n|$)", rev_section)
    for rev_num, date, desc, author in rev_matches:
        revision_history.append({
            "revision": rev_num, "date": date,
            "description": desc.strip(), "author": author.strip(),
        })

    # ── Document Approval ──
    approvals: list[dict[str, str]] = []
    approval_section = _section_text(text, r"DOCUMENT APPROVAL\s*\n", r"$")
    approval_section = re.sub(r"Role\s*\n\s*Name\s*\n\s*Signature\s*\n\s*Date\s*\n?", "", approval_section)
    # Remove page footers
    approval_section = re.sub(r"Page \d+ of \d+\n.*?CONFIDENTIAL.*?\n?", "", approval_section)
    approval_roles = ["Prepared By", "Reviewed By", "Approved By", "Quality Assurance"]
    for role in approval_roles:
        # Name may span multiple lines (e.g. "Dr. Ananya Sharma (QC\nManager)")
        # Greedily capture lines between role and date
        m = re.search(
            re.escape(role) + r"\s*\n((?:.+\n)*?)\s*(\d{2}-\w{3}-\d{4})",
            approval_section
        )
        if m:
            name_text = re.sub(r"\s*\n\s*", " ", m.group(1)).strip()
            approvals.append({
                "role": role,
                "name": name_text,
                "date": m.group(2),
            })

    return {
        "document_id": doc_id,
        "title": title,
        "effective_date": effective_date,
        "revision": revision,
        "department": department,
        "applicable_to": applicable_to,
        "supersedes": supersedes,
        "review_frequency": review_frequency,
        "objective": objective,
        "scope": {"description": scope_intro, "items": scope_items},
        "responsibilities": responsibilities,
        "definitions": definitions,
        "environmental_controls": environmental,
        "product_specifications": specifications,
        "reagents_and_solvents": reagents,
        "equipment": equipment,
        "chromatographic_conditions": chromatographic_conditions,
        "system_suitability": system_suitability,
        "sample_preparation": sample_prep_steps,
        "standard_preparation": standard_prep_steps,
        "calculation": {"formula": calc_formula, "variables": calc_variables},
        "data_integrity": {"description": di_intro, "requirements": data_integrity_items},
        "oos_handling": {"description": oos_intro, "steps": oos_steps},
        "references": references,
        "revision_history": revision_history,
        "approvals": approvals,
    }


def _extract_numbered_steps(text: str, start: str, stop: str) -> list[str]:
    """Extract 'Step N:' or '1. ...' numbered steps from a section."""
    section = _section_text(text, start, stop)
    steps: list[str] = []
    current = ""
    for line in section.splitlines():
        line = line.strip()
        # Check if line starts a new step
        m = re.match(r"^(?:Step\s+\d+:|(\d+)\.\s+)", line)
        if m:
            if current:
                steps.append(current)
            # Remove the step prefix
            current = re.sub(r"^(?:Step\s+\d+:\s*|\d+\.\s+)", "", line)
        elif current and line:
            current += " " + line
    if current:
        steps.append(current)
    return steps


# ---------------------------------------------------------------------------
# OOS Extraction
# ---------------------------------------------------------------------------

def extract_oos_data(pdf_path: str | Path) -> dict[str, Any]:
    """Extract structured data from an OOS Investigation Report PDF."""
    text = _read_pdf_text(pdf_path)

    # ── Metadata (newline-separated field/value) ──
    inv_number = _nl_field(text, "Investigation Number")
    product_name = _nl_field(text, "Product Name")
    batch_number = _nl_field(text, "Batch Number")
    mfg_date = _nl_field(text, "Manufacturing Date")
    expiry_date = _nl_field(text, "Expiry Date")
    batch_size = _nl_field(text, "Batch Size")
    analyst = _nl_field(text, "Analyst ID")
    hplc_system = _nl_field(text, "HPLC System")
    date_initiated = _nl_field(text, "Date Initiated")
    sample_type = _nl_field(text, "Sample Type")
    test_performed = _nl_field(text, "Test Performed")

    # ── Section I: OOS Event ──
    # Table: Parameter\nSpecification\nObserved Result\nStatus
    oos_events: list[dict[str, Any]] = []
    event_rows = re.findall(
        r"(Assay\s*\([\w]+\)|Dissolution|Hardness|Friability|Weight Variation|Disintegration|Related Substance|Microbial)"
        r"\s*\n\s*([\d.]+%?\s*[-–]\s*[\d.]+%?|NMT\s+[\d.]+%?|NLT\s+[\d.]+%?)"
        r"\s*\n\s*([\d.]+%?)"
        r"\s*\n\s*(OOS\s*[-–]\s*\w+|\w+)",
        text, re.IGNORECASE
    )
    for row in event_rows:
        param, spec, observed, status = [r.strip() for r in row]
        oos_events.append({
            "parameter": param,
            "specification": spec,
            "observed_result": observed,
            "status": status,
        })

    # Also extract the narrative description of the OOS event
    oos_description = ""
    desc_match = re.search(
        r"Description of OOS Event\s*\n(.+?)(?:\n\s*Parameter\s*\n|\n\d+\.\d+)",
        text, re.DOTALL
    )
    if desc_match:
        oos_description = re.sub(r"\s+", " ", desc_match.group(1)).strip()

    # ── Initial Actions Taken (Section 1.2) ──
    initial_actions = _extract_bullets(
        text, r"Initial\s+Actions\s+Taken", r"(?:SECTION\s+II|PHASE\s+I|2\.0\s)"
    )

    # ── Section II: Phase I - Laboratory Investigation ──

    # Analyst interview summary
    analyst_interview = ""
    ai_match = re.search(
        r"Analyst Interview Summary\s*\n(.+?)(?:\n\d+\.\d+|\nLaboratory Investigation)",
        text, re.DOTALL
    )
    if ai_match:
        analyst_interview = re.sub(r"\s+", " ", ai_match.group(1)).strip()

    # Lab Investigation Checklist
    # Format: Area\nItem\nFinding\n✓ Status
    checklist: list[dict[str, str]] = []
    checklist_areas = [
        "Sample", "Glassware", "Reagents", "Standard",
        "Data", "Instrument", "Procedure", "Training",
    ]
    checklist_pattern = (
        r"(" + "|".join(checklist_areas) + r")"
        r"\s*\n\s*(.+?)"
        r"\s*\n\s*(.+?)"
        r"\s*\n\s*[✓]?\s*(OK|FAIL|N/A|Pass)"
    )
    checklist_rows = re.findall(checklist_pattern, text, re.IGNORECASE)
    for area, item, finding, status in checklist_rows:
        checklist.append({
            "area": area.strip(),
            "item": item.strip(),
            "finding": finding.strip(),
            "status": status.strip(),
        })

    # System suitability values (embedded in checklist finding)
    system_suitability: dict[str, float] = {}
    ss_match = re.search(r"TF:\s*([\d.]+);\s*Plates:\s*([\d.]+);\s*RSD:\s*([\d.]+)%", text)
    if ss_match:
        system_suitability = {
            "tailing_factor": float(ss_match.group(1)),
            "theoretical_plates": float(ss_match.group(2)),
            "rsd_percent": float(ss_match.group(3)),
        }

    # Phase I conclusion
    phase1_conclusion = ""
    p1_match = re.search(r"PHASE I CONCLUSION:\s*(.+?)(?:\n|$)", text)
    if p1_match:
        phase1_conclusion = p1_match.group(1).strip()

    phase1_recommendation = ""
    p1r_match = re.search(r"RECOMMENDATION:\s*(.+?)(?:\n|$)", text)
    if p1r_match:
        phase1_recommendation = p1r_match.group(1).strip()

    # ── Section III: Phase II - Manufacturing Investigation ──

    # Investigation team
    team_section = _section_text(text, r"Investigation Team\s*\n", r"3\.2\s+Batch Manufacturing")
    team_section = re.sub(r"Name\s*\n\s*Department\s*\n\s*Role\s*\n?", "", team_section)
    investigation_team: list[dict[str, str]] = []
    # Team rows are: Name\nDepartment\nRole (3 lines each)
    team_names = [
        "Dr. Ananya Sharma", "Mr. Vikram Singh", "Mr. Arun Mehta",
        "Ms. Deepa Nair", "Dr. Rahul Kapoor",
    ]
    team_rows = _parse_newline_table(team_section, team_names, 2)
    for row in team_rows:
        vals = row["values"]
        # Clean R&D; artifact from PDF
        dept = vals[0].rstrip(";") if vals else ""
        role = vals[1] if len(vals) > 1 else ""
        investigation_team.append({
            "name": row["name"],
            "department": dept,
            "role": role,
        })

    # BMR Review
    bmr_steps: list[dict[str, str]] = []
    bmr_step_names = [
        "API Weight", "Excipients", "Granulation Moisture",
        "Drying Time", "Blending Time", "Blender Speed", "Compression Force",
    ]
    bmr_section = _section_text(text, r"Batch Manufacturing Record.*?Review\s*\n", r"3\.3\s+Equipment Investigation")
    bmr_section = re.sub(r"Step\s*\n\s*Specification\s*\n\s*Actual\s*\n\s*Status\s*\n?", "", bmr_section)
    bmr_pattern = (
        r"(" + "|".join(re.escape(s) for s in bmr_step_names) + r")"
        r"\s*\n\s*(.+?)"
        r"\s*\n\s*(.+?)"
        r"\s*\n\s*[✓]?\s*(Pass|Fail)"
    )
    bmr_rows = re.findall(bmr_pattern, bmr_section if bmr_section else text, re.IGNORECASE)
    for step, spec, actual, status in bmr_rows:
        bmr_steps.append({
            "step": step.strip(),
            "specification": spec.strip(),
            "actual": actual.strip(),
            "status": status.strip(),
        })

    # Equipment investigation (Section 3.3)
    # Bullets use \x7f. Stop at next subsection (3.4 Root Cause Analysis).
    equip_section = _section_text(
        text, r"3\.3\s+Equipment Investigation.*?\n", r"3\.4\s+Root Cause"
    )
    equipment_findings: list[str] = []
    for line in equip_section.splitlines():
        line = line.strip().lstrip("\x7f\u2022•-").strip()
        if line and re.match(r"^(Visual|Maintenance|Historical|CRITICAL)", line, re.IGNORECASE):
            equipment_findings.append(line)

    # ── Root Cause Analysis ──

    # 5-Whys (newline-separated: Level\nQuestion\nAnswer which may span lines)
    five_whys: list[dict[str, str]] = []
    whys_section = _section_text(text, r"Root Cause Analysis.*?5 Whys\s*\n", r"(?:CONFIRMED\s+ROOT|PHARMAGEN|Page \d)")
    whys_section = re.sub(r"Level\s*\n\s*Question\s*\n\s*Answer\s*\n?", "", whys_section)
    # Parse: Why N\nQuestion\nAnswer (answer may wrap to next line before next Why)
    why_blocks = re.split(r"(?=Why\s+\d+\s*\n)", whys_section)
    for block in why_blocks:
        block = block.strip()
        if not block:
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) >= 3:
            level = lines[0]
            question = lines[1]
            answer = " ".join(lines[2:])  # Join multi-line answers
            five_whys.append({
                "level": level,
                "question": question,
                "answer": answer,
            })

    # Confirmed root cause
    root_cause = ""
    rc_match = re.search(
        r"CONFIRMED\s+ROOT\s+CAUSE:\s*(.+?)(?:\nPHARMAGEN|\nSECTION|\n\n|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if rc_match:
        root_cause = re.sub(r"\s+", " ", rc_match.group(1)).strip()

    # ── Section IV: Risk Assessment ──
    risk_items: list[dict[str, str]] = []
    risk_categories = ["Patient Safety", "Product Quality", "Other Batches", "Regulatory"]
    risk_pattern = (
        r"(" + "|".join(re.escape(c) for c in risk_categories) + r")"
        r"\s*\n\s*(.+?)"
        r"\s*\n\s*(HIGH|MEDIUM|LOW)"
    )
    risk_rows = re.findall(risk_pattern, text, re.IGNORECASE)
    for category, assessment, severity in risk_rows:
        risk_items.append({
            "category": category.strip(),
            "assessment": assessment.strip(),
            "severity": severity.strip().upper(),
        })

    # ── Section V: Batch Disposition ──
    batch_disposition = "UNKNOWN"
    if re.search(r"BATCH.*?IS\s+REJECTED", text, re.IGNORECASE):
        batch_disposition = "REJECTED"
    elif re.search(r"BATCH.*?IS\s+APPROVED", text, re.IGNORECASE):
        batch_disposition = "APPROVED"

    disposition_details = ""
    dd_match = re.search(
        r"IS\s+REJECTED\s*\n(.+?)(?:\nSECTION|\nCAPA)", text, re.DOTALL | re.IGNORECASE
    )
    if dd_match:
        disposition_details = re.sub(r"\s+", " ", dd_match.group(1)).strip()

    # ── Section VI: CAPA Actions ──
    capa_ref = _re_first(r"CAPA\s+Reference:\s*([\w-]+)", text) or ""

    # CAPA table: #\nType\nDescription\nOwner\nDue Date
    # Owner field may contain R&D;/QA (HTML entity artifact) so be flexible
    capa_actions: list[dict[str, str]] = []
    capa_section = _section_text(text, r"CAPA\s+Reference:.*?\n", r"(?:SECTION\s+VII|INVESTIGATION CONCLUSION)")
    # Remove table header
    capa_section = re.sub(r"#\s*\n\s*Type\s*\n\s*Description\s*\n\s*Owner\s*\n\s*Due Date\s*\n?", "", capa_section)
    # Parse each CAPA entry
    capa_blocks = re.split(r"(?=(?:^|\n)\d+\s*\n\s*(?:Corrective|Preventive))", capa_section)
    for block in capa_blocks:
        block = block.strip()
        if not block:
            continue
        m = re.match(
            r"(\d+)\s*\n\s*(Corrective|Preventive)\s*\n\s*(.+?)\s*\n\s*(.+?)\s*\n\s*(\d{2}-\w{3}-\d{4})",
            block, re.IGNORECASE | re.DOTALL
        )
        if m:
            owner = m.group(4).strip().rstrip(";").replace("R&D;", "R&D")
            capa_actions.append({
                "number": m.group(1).strip(),
                "type": m.group(2).strip(),
                "description": re.sub(r"\s+", " ", m.group(3)).strip(),
                "owner": owner,
                "due_date": m.group(5).strip(),
            })

    # ── Section VII: Investigation Conclusion ──
    conclusion = ""
    conc_match = re.search(
        r"INVESTIGATION CONCLUSION\s*\n(.+?)(?:INVESTIGATION APPROVAL|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if conc_match:
        conclusion = re.sub(r"\s+", " ", conc_match.group(1)).strip()

    # Investigation status (usually at end of conclusion)
    investigation_status = ""
    status_match = re.search(r"Investigation status:\s*(\w+)", text, re.IGNORECASE)
    if status_match:
        investigation_status = status_match.group(1).strip().upper()

    # ── Investigation Approval ──
    approvals: list[dict[str, str]] = []
    approval_section = _section_text(text, r"INVESTIGATION APPROVAL\s*\n", r"(?:DISTRIBUTION|$)")
    approval_section = re.sub(r"Role\s*\n\s*Name\s*\n\s*Signature\s*\n\s*Date\s*\n?", "", approval_section)
    approval_roles = ["Prepared By", "Reviewed By", "Mfg. Review", "Final Approval"]
    for role in approval_roles:
        m = re.search(
            re.escape(role) + r"\s*\n\s*(.+?(?:\n.+?)?)\s*\n\s*(\d{2}-\w{3}-\d{4})",
            approval_section
        )
        if m:
            name = re.sub(r"\s*\n\s*", " ", m.group(1)).strip()
            approvals.append({"role": role, "name": name, "date": m.group(2)})

    # ── Distribution ──
    distribution: list[str] = []
    dist_match = re.search(r"DISTRIBUTION\s*\n(.+?)(?:\Z)", text, re.DOTALL)
    if dist_match:
        raw = dist_match.group(1).replace("\x7f", "|").replace("\n", " ")
        distribution = [d.strip() for d in raw.split("|") if d.strip()]

    return {
        "investigation_number": inv_number,
        "product_name": product_name,
        "batch_number": batch_number,
        "manufacturing_date": mfg_date,
        "expiry_date": expiry_date,
        "batch_size": batch_size,
        "sample_type": sample_type,
        "test_performed": test_performed,
        "analyst": analyst,
        "hplc_system": hplc_system,
        "date_initiated": date_initiated,
        "oos_description": oos_description,
        "oos_events": oos_events,
        "initial_actions": initial_actions,
        "analyst_interview": analyst_interview,
        "lab_investigation_checklist": checklist,
        "system_suitability": system_suitability,
        "phase1_conclusion": phase1_conclusion,
        "phase1_recommendation": phase1_recommendation,
        "investigation_team": investigation_team,
        "bmr_review": bmr_steps,
        "equipment_findings": equipment_findings,
        "five_whys": five_whys,
        "root_cause": root_cause,
        "risk_assessment": risk_items,
        "batch_disposition": batch_disposition,
        "disposition_details": disposition_details,
        "capa_reference": capa_ref,
        "capa_actions": capa_actions,
        "conclusion": conclusion,
        "investigation_status": investigation_status,
        "approvals": approvals,
        "distribution": distribution,
    }


# ---------------------------------------------------------------------------
# LLM-Enhanced Extraction
# ---------------------------------------------------------------------------

_SOP_EXTRACTION_PROMPT = """\
You are a pharmaceutical document analyst. Extract ALL structured data from this SOP (Standard Operating Procedure) PDF text.

Return a JSON object with these keys (use null for missing fields, empty arrays for missing lists):

{
  "document_id": "string",
  "title": "string",
  "effective_date": "string",
  "revision": "string",
  "department": "string",
  "applicable_to": "string",
  "supersedes": "string",
  "review_frequency": "string",
  "objective": "full objective text",
  "scope": {"description": "intro text", "items": ["item1", "item2"]},
  "responsibilities": [{"role": "QC Analyst", "responsibilities": "full text"}],
  "definitions": [{"term": "API", "definition": "Active Pharmaceutical Ingredient"}],
  "environmental_controls": [{"parameter": "Temperature", "specification": "22°C - 25°C", "monitoring_frequency": "Continuous", "lower_bound": 22.0, "upper_bound": 25.0}],
  "product_specifications": [{"parameter": "Assay (Paracetamol)", "acceptance_criteria": "95.0% - 105.0% of labeled amount", "reference": "BP", "test_method": "HPLC", "lower_bound": 95.0, "upper_bound": 105.0, "unit": "%", "criteria_type": "range"}],
  "reagents_and_solvents": [{"item": "Methanol", "grade": "HPLC Grade", "storage": "Flammable cabinet, RT"}],
  "equipment": [{"equipment": "HPLC System", "specification": "...", "calibration_frequency": "Annual"}],
  "chromatographic_conditions": [{"parameter": "Column", "specification": "C18, 250mm x 4.6mm, 5µm"}],
  "system_suitability": [{"parameter": "Tailing Factor", "acceptance_criteria": "NMT 2.0", "lower_bound": null, "upper_bound": 2.0, "unit": "", "criteria_type": "nmt"}],
  "sample_preparation": ["step 1 text", "step 2 text"],
  "standard_preparation": ["step 1 text", "step 2 text"],
  "calculation": {"formula": "% Label Claim = ...", "variables": [{"symbol": "Asample", "description": "Peak area of sample"}]},
  "data_integrity": {"description": "intro text about ALCOA+", "requirements": ["requirement 1", "requirement 2"]},
  "oos_handling": {"description": "intro text", "steps": ["STOP testing immediately", "Retain all samples"]},
  "references": ["British Pharmacopoeia - Paracetamol Tablets Monograph", "..."],
  "revision_history": [{"revision": "01", "date": "25-Feb-2026", "description": "Initial Release", "author": "Dr. S. Patel"}],
  "approvals": [{"role": "Prepared By", "name": "Dr. Sunita Patel (QC Analyst)", "date": "25-Feb-2026"}]
}

For criteria_type use: "range" (X-Y), "nmt" (Not More Than), "nlt" (Not Less Than), "pct_deviation" (±X%), "qualitative" (text only).
Parse numeric bounds where possible. Extract EVERY section completely.

PDF TEXT:
"""

_OOS_EXTRACTION_PROMPT = """\
You are a pharmaceutical document analyst. Extract ALL structured data from this OOS (Out-of-Specification) Investigation Report PDF text.

Return a JSON object with these keys (use null for missing fields, empty arrays for missing lists):

{
  "investigation_number": "string",
  "product_name": "string",
  "batch_number": "string",
  "manufacturing_date": "string",
  "expiry_date": "string",
  "batch_size": "string",
  "sample_type": "string",
  "test_performed": "string",
  "analyst": "string",
  "hplc_system": "string",
  "date_initiated": "string",
  "oos_description": "full narrative description of the OOS event",
  "oos_events": [{"parameter": "Assay (Paracetamol)", "specification": "95.0% - 105.0%", "observed_result": "88.0%", "status": "OOS - FAIL"}],
  "initial_actions": ["Testing was immediately halted...", "Analyst notified supervisor..."],
  "analyst_interview": "summary text",
  "lab_investigation_checklist": [{"area": "Sample", "item": "Seal and container", "finding": "Intact", "status": "OK"}],
  "system_suitability": {"tailing_factor": 1.4, "theoretical_plates": 4521, "rsd_percent": 0.8},
  "phase1_conclusion": "string",
  "phase1_recommendation": "string",
  "investigation_team": [{"name": "Dr. Ananya Sharma", "department": "Quality Control", "role": "Lead Investigator"}],
  "bmr_review": [{"step": "API Weight", "specification": "125.00 kg ± 0.5%", "actual": "125.02 kg", "status": "Pass"}],
  "equipment_findings": ["Visual Inspection: Minor wear...", "CRITICAL FINDING: dead spot..."],
  "five_whys": [{"level": "Why 1", "question": "Why was the assay result low?", "answer": "Sample had lower content"}],
  "root_cause": "full confirmed root cause text",
  "risk_assessment": [{"category": "Patient Safety", "assessment": "Sub-potent tablets...", "severity": "HIGH"}],
  "batch_disposition": "REJECTED or APPROVED",
  "disposition_details": "destruction details text",
  "capa_reference": "CAPA-2026-0089",
  "capa_actions": [{"number": "1", "type": "Corrective", "description": "...", "owner": "Engineering", "due_date": "28-Feb-2026"}],
  "conclusion": "full conclusion text",
  "investigation_status": "CLOSED",
  "approvals": [{"role": "Prepared By", "name": "Mr. Rajesh Kumar (QC Supervisor)", "date": "25-Feb-2026"}],
  "distribution": ["Quality Assurance (Original)", "Quality Control (Copy)"]
}

Extract EVERY section completely. Include full text for narrative fields.

PDF TEXT:
"""


def _call_openai(prompt: str, text: str) -> dict[str, Any]:
    """Call OpenAI API to extract structured data from PDF text."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package required for LLM extraction. Install with: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise pharmaceutical document extraction system. Always return valid JSON."},
            {"role": "user", "content": prompt + text},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    return _json.loads(content)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base. Override wins for non-empty values."""
    result = dict(base)
    for key, val in override.items():
        if key not in result:
            result[key] = val
            continue
        base_val = result[key]
        # Override wins if base is empty/None and override has content
        if _is_empty(base_val) and not _is_empty(val):
            result[key] = val
        elif isinstance(base_val, dict) and isinstance(val, dict):
            result[key] = _deep_merge(base_val, val)
        elif isinstance(base_val, list) and isinstance(val, list):
            # Use whichever list is longer/more complete
            if len(val) > len(base_val):
                result[key] = val
    return result


def _is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, (list, dict)) and not val:
        return True
    return False


def extract_sop_data_llm(pdf_path: str | Path) -> dict[str, Any]:
    """Extract SOP data using LLM, merged with regex extraction for robustness."""
    text = _read_pdf_text(pdf_path)

    # Get regex-based extraction as baseline
    regex_result = extract_sop_data(pdf_path)

    # Get LLM extraction
    llm_result = _call_openai(_SOP_EXTRACTION_PROMPT, text)

    # Merge: regex is baseline, LLM fills gaps
    return _deep_merge(regex_result, llm_result)


def extract_oos_data_llm(pdf_path: str | Path) -> dict[str, Any]:
    """Extract OOS data using LLM, merged with regex extraction for robustness."""
    text = _read_pdf_text(pdf_path)

    # Get regex-based extraction as baseline
    regex_result = extract_oos_data(pdf_path)

    # Get LLM extraction
    llm_result = _call_openai(_OOS_EXTRACTION_PROMPT, text)

    # Merge: regex is baseline, LLM fills gaps
    return _deep_merge(regex_result, llm_result)


def extract_sop_data_llm_only(pdf_path: str | Path) -> dict[str, Any]:
    """Extract SOP data using LLM only — no regex/static rules."""
    text = _read_pdf_text(pdf_path)
    return _call_openai(_SOP_EXTRACTION_PROMPT, text)


def extract_oos_data_llm_only(pdf_path: str | Path) -> dict[str, Any]:
    """Extract OOS data using LLM only — no regex/static rules."""
    text = _read_pdf_text(pdf_path)
    return _call_openai(_OOS_EXTRACTION_PROMPT, text)


# ---------------------------------------------------------------------------
# Comparison Engine
# ---------------------------------------------------------------------------

def _normalize_param(name: str) -> str:
    """Normalize parameter name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _safe_dict(value: Any) -> dict[str, Any]:
    """Return a dict or a safe empty fallback."""
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    """Return a list or a safe empty fallback."""
    return value if isinstance(value, list) else []


def _safe_str(value: Any) -> str:
    """Return a string or a safe empty fallback."""
    if value is None:
        return ""
    return str(value)


def _parse_numeric(value_str: str) -> float | None:
    """Extract the first number from a result string like '88.0%'."""
    m = re.search(r"([\d.]+)", str(value_str))
    return float(m.group(1)) if m else None


def _check_compliance(
    observed: float,
    lower: float | None,
    upper: float | None,
    criteria_type: str | None,
) -> tuple[str, str]:
    """Check if observed value meets acceptance criteria.

    Returns (status, deviation_description).
    """
    if criteria_type == "range":
        if lower is not None and observed < lower:
            return "non_compliant", f"-{lower - observed:.1f} below lower limit ({lower})"
        if upper is not None and observed > upper:
            return "non_compliant", f"+{observed - upper:.1f} above upper limit ({upper})"
        return "compliant", ""
    elif criteria_type == "nmt":
        if upper is not None and observed > upper:
            return "non_compliant", f"+{observed - upper:.1f} above limit ({upper})"
        return "compliant", ""
    elif criteria_type == "nlt":
        if lower is not None and observed < lower:
            return "non_compliant", f"-{lower - observed:.1f} below limit ({lower})"
        return "compliant", ""
    return "compliant", ""


def compare_sop_oos(sop_data: dict, oos_data: dict) -> dict[str, Any]:
    """Compare SOP specifications against OOS investigation results."""
    sop_data = _safe_dict(sop_data)
    oos_data = _safe_dict(oos_data)

    # Build lookup of OOS events by normalized parameter name
    oos_lookup: dict[str, dict] = {}
    for event in _safe_list(oos_data.get("oos_events")):
        if not isinstance(event, dict):
            continue
        key = _normalize_param(_safe_str(event.get("parameter")))
        if not key:
            continue
        oos_lookup[key] = event

    # --- Product Specification Results ---
    spec_results: list[dict[str, Any]] = []
    compliant_count = 0
    non_compliant_count = 0
    not_tested_count = 0

    for spec in _safe_list(sop_data.get("product_specifications")):
        if not isinstance(spec, dict):
            continue
        param_name = _safe_str(spec.get("parameter"))
        if not param_name:
            continue
        norm_key = _normalize_param(param_name)

        # Try to find matching OOS event
        oos_event = oos_lookup.get(norm_key)
        if not oos_event:
            for k, v in oos_lookup.items():
                if k in norm_key or norm_key in k:
                    oos_event = v
                    break

        if oos_event:
            observed_result = _safe_str(oos_event.get("observed_result"))
            observed_val = _parse_numeric(observed_result)
            if observed_val is not None:
                status, deviation = _check_compliance(
                    observed_val,
                    spec.get("lower_bound"),
                    spec.get("upper_bound"),
                    spec.get("criteria_type"),
                )
            else:
                status = "not_tested"
                deviation = "Could not parse observed value"

            if status == "compliant":
                compliant_count += 1
            elif status == "non_compliant":
                non_compliant_count += 1
            else:
                not_tested_count += 1

            spec_results.append({
                "parameter": param_name,
                "sop_criteria": _safe_str(spec.get("acceptance_criteria")),
                "reference": spec.get("reference", ""),
                "test_method": spec.get("test_method", ""),
                "oos_result": observed_result,
                "status": status,
                "deviation": deviation,
            })
        else:
            not_tested_count += 1
            spec_results.append({
                "parameter": param_name,
                "sop_criteria": _safe_str(spec.get("acceptance_criteria")),
                "reference": spec.get("reference", ""),
                "test_method": spec.get("test_method", ""),
                "oos_result": None,
                "status": "not_tested",
                "deviation": "Parameter not tested in OOS report",
            })

    total_params = len(spec_results)
    compliance_rate = (compliant_count / total_params * 100) if total_params else 0.0

    # --- System Suitability Results ---
    ss_results: list[dict[str, Any]] = []
    oos_ss = _safe_dict(oos_data.get("system_suitability"))

    ss_mapping = {
        "Tailing Factor": ("tailing_factor", oos_ss.get("tailing_factor")),
        "Theoretical Plates": ("theoretical_plates", oos_ss.get("theoretical_plates")),
        "%RSD of Peak Area": ("rsd_percent", oos_ss.get("rsd_percent")),
    }

    for sop_ss in _safe_list(sop_data.get("system_suitability")):
        if not isinstance(sop_ss, dict):
            continue
        param = _safe_str(sop_ss.get("parameter"))
        if not param:
            continue
        mapping = ss_mapping.get(param)
        if mapping and mapping[1] is not None:
            observed = mapping[1]
            status, deviation = _check_compliance(
                observed,
                sop_ss.get("lower_bound"),
                sop_ss.get("upper_bound"),
                sop_ss.get("criteria_type"),
            )
            ss_results.append({
                "parameter": param,
                "sop_criteria": _safe_str(sop_ss.get("acceptance_criteria")),
                "oos_result": observed,
                "status": status,
                "deviation": deviation,
            })
        else:
            ss_results.append({
                "parameter": param,
                "sop_criteria": _safe_str(sop_ss.get("acceptance_criteria")),
                "oos_result": None,
                "status": "not_tested",
                "deviation": "Not reported in OOS investigation",
            })

    # --- OOS Procedure Compliance ---
    procedure_compliance: list[dict[str, Any]] = []
    initial_actions = [_safe_str(item) for item in _safe_list(oos_data.get("initial_actions")) if item is not None]
    actions_text = " ".join(initial_actions).lower()

    # Map each SOP OOS step to keywords to look for in OOS initial actions
    step_checks = [
        ("STOP testing immediately", ["halt", "stop", "testing was immediately halted"]),
        ("Retain all sample solutions, standards, and reagents",
         ["retain", "sample solutions", "retained and secured"]),
        ("Secure all equipment and vials for supervisor review",
         ["equipment", "secured", "hold", "placed on hold"]),
        ("Notify QC Supervisor within 30 minutes",
         ["notified", "supervisor", "within 15 minutes", "within 30 minutes"]),
        ("Complete OOS Notification Form (QC-FORM-OOS-01)",
         ["oos notification form", "qc-form-oos"]),
        ("Do not discard any materials until Phase I investigation complete",
         ["not discard", "retained", "secured", "chain of custody"]),
    ]

    for step_desc, keywords in step_checks:
        found = any(kw in actions_text for kw in keywords)
        # Try to find the specific evidence line
        evidence = ""
        if found:
            for action in initial_actions:
                if any(kw in action.lower() for kw in keywords):
                    evidence = action
                    break
        procedure_compliance.append({
            "sop_requirement": step_desc,
            "status": "compliant" if found else "not_verified",
            "evidence": evidence if evidence else ("Not explicitly documented" if not found else "Found in initial actions"),
        })

    # --- Overall compliance rate across all visible checks ---
    total_spec_checks = len(spec_results)
    total_ss_checks = len(ss_results)
    total_procedure_checks = len(procedure_compliance)
    total_checks = total_spec_checks + total_ss_checks + total_procedure_checks

    compliant_spec_checks = sum(1 for item in spec_results if item.get("status") == "compliant")
    compliant_ss_checks = sum(1 for item in ss_results if item.get("status") == "compliant")
    compliant_procedure_checks = sum(1 for item in procedure_compliance if item.get("status") == "compliant")
    compliant_checks = compliant_spec_checks + compliant_ss_checks + compliant_procedure_checks

    compliance_rate = (compliant_checks / total_checks * 100) if total_checks else 0.0

    # --- Build output ---
    return {
        "sop_document": {
            "id": _safe_str(sop_data.get("document_id")),
            "title": _safe_str(sop_data.get("title")),
            "effective_date": _safe_str(sop_data.get("effective_date")),
            "revision": _safe_str(sop_data.get("revision")),
            "department": _safe_str(sop_data.get("department")),
            "applicable_to": _safe_str(sop_data.get("applicable_to")),
        },
        "oos_document": {
            "id": _safe_str(oos_data.get("investigation_number")),
            "batch": _safe_str(oos_data.get("batch_number")),
            "product": _safe_str(oos_data.get("product_name")),
            "date_initiated": _safe_str(oos_data.get("date_initiated")),
            "analyst": _safe_str(oos_data.get("analyst")),
            "hplc_system": _safe_str(oos_data.get("hplc_system")),
            "manufacturing_date": _safe_str(oos_data.get("manufacturing_date")),
            "expiry_date": _safe_str(oos_data.get("expiry_date")),
            "batch_size": _safe_str(oos_data.get("batch_size")),
            "sample_type": _safe_str(oos_data.get("sample_type")),
            "test_performed": _safe_str(oos_data.get("test_performed")),
        },
        "product_specification_results": spec_results,
        "system_suitability_results": ss_results,
        "oos_procedure_compliance": procedure_compliance,
        "investigation_summary": {
            "oos_description": _safe_str(oos_data.get("oos_description")),
            "analyst_interview": _safe_str(oos_data.get("analyst_interview")),
            "phase1_conclusion": _safe_str(oos_data.get("phase1_conclusion")),
            "phase1_recommendation": _safe_str(oos_data.get("phase1_recommendation")),
            "investigation_team": _safe_list(oos_data.get("investigation_team")),
            "root_cause": _safe_str(oos_data.get("root_cause")),
            "five_whys": _safe_list(oos_data.get("five_whys")),
            "batch_disposition": _safe_str(oos_data.get("batch_disposition")),
            "disposition_details": _safe_str(oos_data.get("disposition_details")),
            "risk_assessment": _safe_list(oos_data.get("risk_assessment")),
            "capa_reference": _safe_str(oos_data.get("capa_reference")),
            "capa_actions": _safe_list(oos_data.get("capa_actions")),
            "equipment_findings": _safe_list(oos_data.get("equipment_findings")),
            "conclusion": _safe_str(oos_data.get("conclusion")),
            "investigation_status": _safe_str(oos_data.get("investigation_status")),
            "bmr_review": _safe_list(oos_data.get("bmr_review")),
            "lab_checklist": _safe_list(oos_data.get("lab_investigation_checklist")),
            "approvals": _safe_list(oos_data.get("approvals")),
            "distribution": _safe_list(oos_data.get("distribution")),
        },
        "sop_details": {
            "objective": _safe_str(sop_data.get("objective")),
            "scope": _safe_dict(sop_data.get("scope")),
            "responsibilities": _safe_list(sop_data.get("responsibilities")),
            "definitions": _safe_list(sop_data.get("definitions")),
            "reagents_and_solvents": _safe_list(sop_data.get("reagents_and_solvents")),
            "equipment": _safe_list(sop_data.get("equipment")),
            "chromatographic_conditions": _safe_list(sop_data.get("chromatographic_conditions")),
            "sample_preparation": _safe_list(sop_data.get("sample_preparation")),
            "standard_preparation": _safe_list(sop_data.get("standard_preparation")),
            "calculation": _safe_dict(sop_data.get("calculation")),
            "data_integrity": _safe_dict(sop_data.get("data_integrity")),
            "references": _safe_list(sop_data.get("references")),
            "revision_history": _safe_list(sop_data.get("revision_history")),
            "approvals": _safe_list(sop_data.get("approvals")),
        },
        "summary": {
            "total_parameters": total_params,
            "compliant": compliant_count,
            "non_compliant": non_compliant_count,
            "not_tested": not_tested_count,
            "total_system_suitability": total_ss_checks,
            "compliant_system_suitability": compliant_ss_checks,
            "total_procedure_checks": total_procedure_checks,
            "verified_procedure_checks": compliant_procedure_checks,
            "not_verified_procedure_checks": total_procedure_checks - compliant_procedure_checks,
            "total_checks": total_checks,
            "compliant_checks": compliant_checks,
            "compliance_rate": round(compliance_rate, 1),
        },
    }
