#!/usr/bin/env python3
"""
SOP vs OOS Compliance Analyzer - Streamlit UI
Compares pharmaceutical SOP acceptance criteria against OOS investigation results.
"""

import json
import sqlite3
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime
from typing import Optional, List

import streamlit as st


API_URL = "http://localhost:8000"
HISTORY_DB_PATH = Path(__file__).resolve().parent / "data" / "sop_oos_history.db"


# =============================================================================
# HISTORY (local SQLite)
# =============================================================================

def init_history_db():
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sop_oos_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sop_name TEXT,
            oos_name TEXT,
            compliance_rate REAL,
            batch_disposition TEXT,
            total_params INTEGER,
            compliant INTEGER,
            non_compliant INTEGER,
            not_tested INTEGER,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_to_history(sop_name: str, oos_name: str, results: dict) -> int:
    conn = sqlite3.connect(HISTORY_DB_PATH)
    summary = _safe_dict(results.get("summary"))
    inv = _safe_dict(results.get("investigation_summary"))
    cur = conn.execute("""
        INSERT INTO sop_oos_history (
            timestamp, sop_name, oos_name, compliance_rate, batch_disposition,
            total_params, compliant, non_compliant, not_tested, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        sop_name, oos_name,
        summary.get("compliance_rate", 0),
        inv.get("batch_disposition", ""),
        summary.get("total_parameters", 0),
        summary.get("compliant", 0),
        summary.get("non_compliant", 0),
        summary.get("not_tested", 0),
        json.dumps(results, default=str),
    ))
    record_id = cur.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_history(limit: int = 50) -> List[dict]:
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM sop_oos_history ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_history_record(record_id: int) -> Optional[dict]:
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM sop_oos_history WHERE id = ?", (record_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_history_record(record_id: int) -> bool:
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.execute("DELETE FROM sop_oos_history WHERE id = ?", (record_id,))
    deleted = conn.total_changes > 0
    conn.commit()
    conn.close()
    return deleted


# =============================================================================
# HTTP HELPERS
# =============================================================================

@st.cache_resource
def get_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def api_get(endpoint: str, timeout: int = 5) -> Optional[dict]:
    try:
        r = get_http_session().get(f"{API_URL}{endpoint}", timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_ok() -> bool:
    return api_get("/") is not None


def _safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_list(value) -> list:
    return value if isinstance(value, list) else []


# =============================================================================
# COLORS / STYLING
# =============================================================================

COLORS = {
    "primary": "#D04A02",
    "secondary": "#515350",
    "dark": "#2B2B2B",
    "success": "#00854D",
    "warning": "#FFA500",
    "danger": "#DC291E",
    "light_bg": "#515350",
    "border": "#DDDDDD",
    "text_secondary": "#666666",
}


def setup_page():
    init_history_db()

    st.set_page_config(
        page_title="SOP vs OOS Compliance Analyzer",
        page_icon="",
        layout="wide",
    )

    st.markdown(f"""
    <style>
        .main .block-container {{
            max-width: 1400px;
            padding-top: 1rem;
            background-color: {COLORS['light_bg']};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, {COLORS['light_bg']} 100%);
            border-right: 1px solid {COLORS['border']};
        }}
        .section-header {{
            color: {COLORS['secondary']};
            font-size: 1.4rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid {COLORS['primary']};
            letter-spacing: -0.01em;
            text-transform: uppercase;
        }}
        .app-header {{
            background: {COLORS['secondary']};
            padding: 2rem 2.5rem;
            border-radius: 0;
            margin-bottom: 2rem;
            border-left: 6px solid {COLORS['primary']};
        }}
        .app-header-title {{
            font-size: 2rem;
            font-weight: 700;
            color: white;
            letter-spacing: -0.02em;
        }}
        .app-header-subtitle {{
            color: #CCCCCC;
            font-size: 1rem;
            margin-top: 0.5rem;
            font-weight: 300;
        }}
        .status-strip {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1rem;
            justify-content: center;
        }}
        .status-chip {{
            background: rgba(255, 255, 255, 0.08);
            color: white;
            padding: 0.45rem 0.8rem;
            border: 1px solid rgba(255, 255, 255, 0.14);
            font-size: 0.82rem;
            letter-spacing: 0.02em;
        }}
        .status-chip strong {{
            color: #FFFFFF;
        }}
        .sidebar-section {{
            background: white;
            border-radius: 0;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid {COLORS['primary']};
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }}
        .sidebar-section-title {{
            font-weight: 700;
            color: {COLORS['secondary']};
            margin-bottom: 1rem;
            font-size: 1.1rem;
            letter-spacing: -0.01em;
            text-transform: uppercase;
        }}
        .sidebar-section p {{
            color: {COLORS['text_secondary']};
            font-size: 0.9rem;
            margin: 0;
            line-height: 1.6;
        }}
        .feature-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #374151;
            font-size: 0.85rem;
            margin: 0.4rem 0;
        }}
        .feature-check {{
            color: {COLORS['success']};
            font-weight: bold;
        }}
        .step-item {{
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            margin: 0.5rem 0;
            font-size: 0.85rem;
            color: #374151;
        }}
        .step-num {{
            background: {COLORS['primary']};
            color: white;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            flex-shrink: 0;
        }}
        .summary-card {{
            background: linear-gradient(180deg, #F4F4F4 0%, #ECECEC 100%);
            border: 1px solid #D7D7D7;
            border-radius: 0;
            padding: 1.75rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
            min-height: 126px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .summary-card:hover {{
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
        }}
        .summary-value {{
            font-size: 2.6rem;
            font-weight: 400;
            margin-bottom: 0.5rem;
            line-height: 1;
        }}
        .summary-label {{
            color: {COLORS['secondary']};
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        .upload-panel {{
            background: linear-gradient(180deg, #111827 0%, #1F2937 100%);
            border: 1px solid #374151;
            border-left: 4px solid {COLORS['primary']};
            padding: 1rem;
            margin-bottom: 1rem;
            min-height: 110px;
        }}
        .upload-panel-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #F9FAFB;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .upload-panel-note {{
            font-size: 0.85rem;
            color: #D1D5DB;
            margin-bottom: 0;
        }}
        .helper-card {{
            background: linear-gradient(90deg, #F7F7F7 0%, #EFEFEF 100%);
            border-left: 4px solid {COLORS['primary']};
            padding: 0.9rem 1rem;
            margin: 0.75rem 0 1rem 0;
            color: {COLORS['secondary']};
        }}
        .helper-card strong {{
            display: block;
            margin-bottom: 0.25rem;
        }}
        .history-card {{
            background: white;
            border: 1px solid {COLORS['border']};
            padding: 1rem 1rem 0.9rem 1rem;
            margin-bottom: 0.9rem;
        }}
        .history-meta {{
            color: {COLORS['text_secondary']};
            font-size: 0.82rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {COLORS['light_bg']};
            padding: 0.5rem;
            border-radius: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            border-radius: 6px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['primary']} !important;
            color: white !important;
        }}
        #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.session_state.results = None
    if "sop_name" not in st.session_state:
        st.session_state.sop_name = None
    if "oos_name" not in st.session_state:
        st.session_state.oos_name = None


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0 1.5rem 0; border-bottom: 2px solid {COLORS['border']};">
            <svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M66.7 47.3001C64.2 47.7001 63 49.5001 63 52.7001C63 55.9001 64.7 58.1001 67.2 58.1001C69.7 58.1001 69.5 57.7001 71.8 56.6001V59.2001C69.1 60.5001 67.5 60.8001 65.2 60.8001C62.9 60.8001 61.1 60.2001 59.8 58.8001C58.4 57.4001 57.7 55.5001 57.7 53.5001C57.7 48.9001 61.1 45.8001 66.1 45.8001C71.1 45.8001 71.7 47.3001 71.7 49.5001C71.7 51.7001 70.6 51.9001 69.1 51.9001C67.6 51.9001 67.6 51.7001 66.8 51.2001V47.3001H66.7ZM54.6 53.4001C56.8 50.6001 57.6 49.5001 57.6 48.1001C57.6 46.7001 56.5 45.6001 55.1 45.6001C53.7 45.6001 53.4 46.0001 53 46.5001V52.2001L49.4 57.0001V46.0001H46L40.3 55.5001V46.0001H38.3L33.1 47.3001V48.6001L35.9 48.9001V60.5001H39.6L45.1 51.5001V60.5001H49.1L54.7 53.4001H54.6ZM22.2 49.0001C23 49.0001 23.5 48.8001 23.9 48.8001C26.3 48.8001 27.6 50.4001 27.6 53.4001C27.6 56.4001 26 58.8001 23.1 58.8001C20.2 58.8001 22.7 58.8001 22.3 58.8001V49.1001L22.2 49.0001ZM22.2 60.4001C23.1 60.4001 24.1 60.4001 24.6 60.4001C29.5 60.4001 32.6 57.3001 32.6 52.6001C32.6 47.9001 30.3 45.7001 27.1 45.7001C23.9 45.7001 24.8 46.0001 22.2 47.6001V45.7001H20.7L15 47.4001V48.8001H17.4V65.0001L15.3 65.5001V66.8001H24.6V65.5001L22.2 65.0001V60.3001V60.4001Z" fill="black"/>
<path d="M64.1 41.8H48.6L51.2 37.4H66.7L64.1 41.8ZM84.9 33H69.4L66.8 37.4H82.3L84.9 33Z" fill="#FD5108"/>
</svg>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">What This Does</div>
            <p>
                Compares your SOP acceptance criteria against OOS investigation
                results. Checks if test values meet specifications and if OOS
                procedures were followed correctly.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Features</div>
            <div class="feature-item"><span class="feature-check">&check;</span> LLM-powered PDF extraction</div>
            <div class="feature-item"><span class="feature-check">&check;</span> Automated compliance checking</div>
            <div class="feature-item"><span class="feature-check">&check;</span> System suitability validation</div>
            <div class="feature-item"><span class="feature-check">&check;</span> OOS procedure compliance audit</div>
            <div class="feature-item"><span class="feature-check">&check;</span> Export to CSV / JSON</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">How To Use</div>
            <div class="step-item"><div class="step-num">1</div> Upload SOP and OOS PDFs</div>
            <div class="step-item"><div class="step-num">2</div> Click Analyze</div>
            <div class="step-item"><div class="step-num">3</div> Review compliance results</div>
        </div>
        """, unsafe_allow_html=True)


def render_header():
    records = get_history(limit=1)
    history_count = len(get_history(limit=200))
    last_run = records[0]["timestamp"][:19].replace("T", " ") if records else "No runs yet"
    server_state = "Online" if api_ok() else "Offline"
    st.markdown("""
    <div class="app-header">
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div class="app-header-title">SOP vs OOS Compliance Analyzer</div>
            <div class="app-header-subtitle">Review specification failures, system suitability, and OOS procedure adherence in one pass</div>
        </div>
        <div class="status-strip">
            <div class="status-chip"><strong>Server:</strong> SERVER_STATE</div>
            <div class="status-chip"><strong>Saved Runs:</strong> HISTORY_COUNT</div>
            <div class="status-chip"><strong>Last Run:</strong> LAST_RUN</div>
        </div>
    </div>
    """.replace("SERVER_STATE", server_state).replace("HISTORY_COUNT", str(history_count)).replace("LAST_RUN", last_run), unsafe_allow_html=True)


def show_connection_error():
    st.error("Cannot connect to the server.")
    st.markdown("""
    **The server needs to be running first.**

    Open a terminal and run:
    ```
    uv run python server.py
    ```
    Then refresh this page.
    """)


# =============================================================================
# UPLOAD & ANALYZE
# =============================================================================

def section_upload_and_analyze():
    st.markdown('<div class="section-header">Upload & Analyze</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="helper-card">
        <strong>Recommended flow</strong>
        Upload the source SOP and the corresponding OOS investigation PDF, choose the extraction mode, then run analysis.
        Use <em>Hybrid</em> if you want a safer fallback when LLM extraction misses fields.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="upload-panel">
            <div class="upload-panel-title">SOP Document</div>
            <div class="upload-panel-note">Standard Operating Procedure PDF with acceptance criteria and OOS handling steps.</div>
        </div>
        """, unsafe_allow_html=True)
        sop_file = st.file_uploader(
            "SOP PDF", type=["pdf"], key="sop_upload", label_visibility="collapsed"
        )
        if sop_file is not None:
            st.caption(f"Loaded: {sop_file.name}")

    with col2:
        st.markdown("""
        <div class="upload-panel">
            <div class="upload-panel-title">OOS Investigation Report</div>
            <div class="upload-panel-note">Investigation PDF with observed results, root cause, and CAPA actions.</div>
        </div>
        """, unsafe_allow_html=True)
        oos_file = st.file_uploader(
            "OOS PDF", type=["pdf"], key="oos_upload", label_visibility="collapsed"
        )
        if oos_file is not None:
            st.caption(f"Loaded: {oos_file.name}")

    st.markdown("---")

    extraction_mode = st.selectbox(
        "Extraction Mode",
        options=["llm", "hybrid", "regex"],
        format_func=lambda x: {
            "llm": "LLM Only (Recommended)",
            "hybrid": "Hybrid (LLM + Regex)",
            "regex": "Regex Only (No LLM)",
        }[x],
        index=0,
        help="LLM Only is fastest to configure but can be less stable on messy PDFs. Hybrid is more defensive.",
    )

    st.caption({
        "llm": "Best for clean PDFs and fastest setup.",
        "hybrid": "Best default for production use: LLM extraction with regex fallback.",
        "regex": "No API use. Most deterministic, but least complete on complex layouts.",
    }[extraction_mode])

    both_ready = sop_file is not None and oos_file is not None

    if st.button("Analyze Compliance", type="primary", disabled=not both_ready, use_container_width=True):
        with st.spinner("Extracting and comparing (LLM mode may take 15-30s)..."):
            try:
                session = get_http_session()
                resp = session.post(
                    f"{API_URL}/sop-oos/analyze",
                    files={
                        "sop_file": (sop_file.name, sop_file.getvalue(), "application/pdf"),
                        "oos_file": (oos_file.name, oos_file.getvalue(), "application/pdf"),
                    },
                    data={"mode": extraction_mode},
                    timeout=120,
                )
                if resp.ok:
                    result = resp.json()
                    st.session_state.results = result
                    st.session_state.sop_name = sop_file.name
                    st.session_state.oos_name = oos_file.name
                    try:
                        save_to_history(sop_file.name, oos_file.name, result)
                    except Exception as e:
                        print(f"History save error: {e}")
                    st.rerun()
                else:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.error(f"Analysis failed: {detail}")
            except requests.ConnectionError:
                show_connection_error()
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# RESULTS
# =============================================================================

def _status_color(status: str) -> str:
    return {
        "compliant": COLORS["success"],
        "non_compliant": COLORS["danger"],
        "not_tested": "#999999",
        "not_verified": COLORS["warning"],
    }.get(status, "#999999")


def _status_label(status: str) -> str:
    return {
        "compliant": "Compliant",
        "non_compliant": "Non-Compliant",
        "not_tested": "Not Tested",
        "not_verified": "Not Verified",
    }.get(status, status.replace("_", " ").title())


def _severity_color(severity: str) -> str:
    return {"HIGH": COLORS["danger"], "MEDIUM": COLORS["warning"], "LOW": COLORS["success"]}.get(severity, "#999")


def section_results():
    r = _safe_dict(st.session_state.results)
    summary = _safe_dict(r.get("summary"))
    sop_doc = _safe_dict(r.get("sop_document"))
    oos_doc = _safe_dict(r.get("oos_document"))
    inv = _safe_dict(r.get("investigation_summary"))

    st.markdown('<div class="section-header">Compliance Results</div>', unsafe_allow_html=True)

    # Start Over button
    col_spacer, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Start Over", use_container_width=True):
            st.session_state.results = None
            st.rerun()

    # --- Executive Summary Cards ---
    compliance_rate = summary.get("compliance_rate", 0)
    disposition = inv.get("batch_disposition", "UNKNOWN")
    non_compliant = summary.get("non_compliant", 0)

    # Risk level from risk assessment
    risk_items = [ri for ri in _safe_list(inv.get("risk_assessment")) if isinstance(ri, dict)]
    max_severity = "LOW"
    for ri in risk_items:
        s = ri.get("severity", "LOW")
        if s == "HIGH":
            max_severity = "HIGH"
            break
        if s == "MEDIUM":
            max_severity = "MEDIUM"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        rate_color = COLORS["success"] if compliance_rate >= 80 else (COLORS["warning"] if compliance_rate >= 50 else COLORS["danger"])
        st.markdown(f"""
        <div class="summary-card" style="border-top: 4px solid {rate_color};">
            <div class="summary-value" style="color: {rate_color};">{compliance_rate}%</div>
            <div class="summary-label">Compliance Rate</div>
        </div>
        """, unsafe_allow_html=True)

        with c2:
            disp_color = COLORS["danger"] if disposition == "REJECTED" else COLORS["success"]
            st.markdown(f"""
            <div class="summary-card" style="border-top: 4px solid {disp_color};">
                <div class="summary-value" style="color: {disp_color};">{disposition}</div>
                <div class="summary-label">Batch Disposition</div>
            </div>
            """, unsafe_allow_html=True)

    with c3:
        nc_color = COLORS["danger"] if non_compliant > 0 else COLORS["success"]
        st.markdown(f"""
        <div class="summary-card" style="border-top: 4px solid {nc_color};">
            <div class="summary-value" style="color: {nc_color};">{non_compliant}</div>
            <div class="summary-label">Non-Compliant Parameters</div>
        </div>
        """, unsafe_allow_html=True)

        with c4:
            risk_color = _severity_color(max_severity)
            st.markdown(f"""
            <div class="summary-card" style="border-top: 4px solid {risk_color};">
                <div class="summary-value" style="color: {risk_color};">{max_severity}</div>
                <div class="summary-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Document Info ---
    with st.expander("Document Information", expanded=False):
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**SOP Document**")
            st.markdown(f"- **ID:** {sop_doc.get('id', '')}")
            st.markdown(f"- **Title:** {sop_doc.get('title', '')}")
            st.markdown(f"- **Effective Date:** {sop_doc.get('effective_date', '')}")
            st.markdown(f"- **Revision:** {sop_doc.get('revision', '')}")
        with dc2:
            st.markdown("**OOS Investigation**")
            st.markdown(f"- **ID:** {oos_doc.get('id', '')}")
            st.markdown(f"- **Batch:** {oos_doc.get('batch', '')}")
            st.markdown(f"- **Product:** {oos_doc.get('product', '')}")
            st.markdown(f"- **Date:** {oos_doc.get('date_initiated', '')}")
            st.markdown(f"- **Analyst:** {oos_doc.get('analyst', '')}")
            st.markdown(f"- **HPLC System:** {oos_doc.get('hplc_system', '')}")

    # --- Product Specification Compliance ---
    st.markdown("### Product Specification Compliance")
    spec_results = [s for s in _safe_list(r.get("product_specification_results")) if isinstance(s, dict)]
    if spec_results:
        rows = []
        for s in spec_results:
            rows.append({
                "Parameter": s.get("parameter", "—"),
                "SOP Criteria": s.get("sop_criteria", "—"),
                "OOS Result": s.get("oos_result") or "—",
                "Status": _status_label(s.get("status", "not_tested")),
                "Deviation": s.get("deviation") or "—",
            })
        df = pd.DataFrame(rows)

        def _color_status(val):
            colors = {
                "Compliant": f"background-color: #d1fae5; color: #065f46",
                "Non-Compliant": f"background-color: #fee2e2; color: #991b1b",
                "Not Tested": f"background-color: #f3f4f6; color: #6b7280",
            }
            return colors.get(val, "")

        styled = df.style.map(_color_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- System Suitability ---
    st.markdown("### System Suitability")
    ss_results = [s for s in _safe_list(r.get("system_suitability_results")) if isinstance(s, dict)]
    if ss_results:
        rows = []
        for s in ss_results:
            rows.append({
                "Parameter": s.get("parameter", "—"),
                "SOP Criteria": s.get("sop_criteria", "—"),
                "OOS Result": s.get("oos_result") if s.get("oos_result") is not None else "—",
                "Status": _status_label(s.get("status", "not_tested")),
                "Deviation": s.get("deviation") or "—",
            })
        df = pd.DataFrame(rows)
        styled = df.style.map(_color_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- OOS Procedure Compliance ---
    st.markdown("### OOS Procedure Compliance")
    proc_results = [p for p in _safe_list(r.get("oos_procedure_compliance")) if isinstance(p, dict)]
    if proc_results:
        for p in proc_results:
            status = p.get("status", "not_verified")
            icon = "&#x2705;" if status == "compliant" else "&#x26A0;&#xFE0F;"
            color = COLORS["success"] if status == "compliant" else COLORS["warning"]
            st.markdown(
                f'<div style="padding: 0.4rem 0; color: {color};">'
                f'{icon} <strong>{p.get("sop_requirement", "Requirement")}</strong> '
                f'<span style="color: #888; font-size: 0.85rem;">({p.get("evidence", "Not explicitly documented")})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- Investigation Summary ---
    st.markdown("### Investigation Summary")

    # OOS Description
    oos_desc = inv.get("oos_description", "")
    if oos_desc:
        with st.expander("OOS Event Description", expanded=False):
            st.markdown(oos_desc)

    # Analyst Interview
    interview = inv.get("analyst_interview", "")
    if interview:
        with st.expander("Analyst Interview", expanded=False):
            st.markdown(interview)

    # Phase 1 Conclusion
    p1_conclusion = inv.get("phase1_conclusion", "")
    p1_recommendation = inv.get("phase1_recommendation", "")
    if p1_conclusion or p1_recommendation:
        with st.expander("Phase I Conclusion", expanded=False):
            if p1_conclusion:
                st.markdown(f"**Conclusion:** {p1_conclusion}")
            if p1_recommendation:
                st.markdown(f"**Recommendation:** {p1_recommendation}")

    # Investigation Team
    team = [item for item in _safe_list(inv.get("investigation_team")) if isinstance(item, dict)]
    if team:
        with st.expander("Investigation Team", expanded=False):
            team_df = pd.DataFrame(team)
            st.dataframe(team_df, use_container_width=True, hide_index=True)

    # Root Cause
    root_cause = inv.get("root_cause", "")
    if root_cause:
        st.info(f"**Root Cause:** {root_cause}")

    # 5-Whys
    five_whys = [item for item in _safe_list(inv.get("five_whys")) if isinstance(item, dict)]
    if five_whys:
        with st.expander("5-Whys Analysis", expanded=False):
            for w in five_whys:
                st.markdown(f"**{w.get('level', '')}:** {w.get('question', '')}")
                st.markdown(f"  {w.get('answer', '')}")

    # Risk Assessment
    risk_items = [ri for ri in _safe_list(inv.get("risk_assessment")) if isinstance(ri, dict)]
    if risk_items:
        with st.expander("Risk Assessment", expanded=False):
            for ri in risk_items:
                color = _severity_color(ri.get("severity", "LOW"))
                st.markdown(
                    f'<span style="color: {color}; font-weight: bold;">[{ri.get("severity", "LOW")}]</span> '
                    f'**{ri.get("category", "Risk")}**: {ri.get("assessment", "")}',
                    unsafe_allow_html=True,
                )

    # Equipment Findings
    equip = [str(item) for item in _safe_list(inv.get("equipment_findings")) if item is not None]
    if equip:
        with st.expander("Equipment Findings", expanded=False):
            for e in equip:
                st.markdown(f"- {e}")

    # BMR Review
    bmr = [item for item in _safe_list(inv.get("bmr_review")) if isinstance(item, dict)]
    if bmr:
        with st.expander("Batch Manufacturing Record", expanded=False):
            bmr_df = pd.DataFrame(bmr)
            st.dataframe(bmr_df, use_container_width=True, hide_index=True)

    # Lab Checklist
    checklist = [item for item in _safe_list(inv.get("lab_checklist")) if isinstance(item, dict)]
    if checklist:
        with st.expander("Lab Investigation Checklist", expanded=False):
            cl_df = pd.DataFrame(checklist)
            st.dataframe(cl_df, use_container_width=True, hide_index=True)

    # CAPA Actions
    capas = [item for item in _safe_list(inv.get("capa_actions")) if isinstance(item, dict)]
    if capas:
        st.markdown("### CAPA Actions")
        capa_df = pd.DataFrame(capas)
        st.dataframe(capa_df, use_container_width=True, hide_index=True)

    # Batch Disposition Details
    disp_details = inv.get("disposition_details", "")
    if disp_details:
        with st.expander("Disposition Details", expanded=False):
            st.markdown(disp_details)

    # Conclusion
    conclusion = inv.get("conclusion", "")
    if conclusion:
        st.markdown("### Conclusion")
        st.markdown(conclusion)

    # Investigation Status & Approvals
    inv_status = inv.get("investigation_status", "")
    inv_approvals = [item for item in _safe_list(inv.get("approvals")) if isinstance(item, dict)]
    if inv_status or inv_approvals:
        with st.expander("Investigation Status & Approvals", expanded=False):
            if inv_status:
                st.markdown(f"**Status:** {inv_status}")
            if inv_approvals:
                approvals_df = pd.DataFrame(inv_approvals)
                st.dataframe(approvals_df, use_container_width=True, hide_index=True)

    # SOP Details
    sop_details = _safe_dict(r.get("sop_details"))
    if sop_details:
        with st.expander("SOP Details", expanded=False):
            obj = sop_details.get("objective", "")
            if obj:
                st.markdown(f"**Objective:** {obj}")

            scope = _safe_dict(sop_details.get("scope"))
            if scope:
                scope_desc = scope.get("description", "")
                if scope_desc:
                    st.markdown(f"**Scope:** {scope_desc}")
                scope_items = _safe_list(scope.get("items"))
                for item in scope_items:
                    st.markdown(f"- {item}")

            responsibilities = [item for item in _safe_list(sop_details.get("responsibilities")) if isinstance(item, dict)]
            if responsibilities:
                st.markdown("**Responsibilities:**")
                resp_df = pd.DataFrame(responsibilities)
                st.dataframe(resp_df, use_container_width=True, hide_index=True)

            chrom = [item for item in _safe_list(sop_details.get("chromatographic_conditions")) if isinstance(item, dict)]
            if chrom:
                st.markdown("**Chromatographic Conditions:**")
                chrom_df = pd.DataFrame(chrom)
                st.dataframe(chrom_df, use_container_width=True, hide_index=True)

            calc = _safe_dict(sop_details.get("calculation"))
            if calc and calc.get("formula"):
                st.markdown(f"**Calculation Formula:** `{calc['formula']}`")
                for v in [item for item in _safe_list(calc.get("variables")) if isinstance(item, dict)]:
                    st.markdown(f"- **{v.get('symbol', '')}**: {v.get('description', '')}")

            data_int = _safe_dict(sop_details.get("data_integrity"))
            if data_int:
                di_desc = data_int.get("description", "")
                if di_desc:
                    st.markdown(f"**Data Integrity:** {di_desc}")

            refs = [str(item) for item in _safe_list(sop_details.get("references")) if item is not None]
            if refs:
                st.markdown("**References:**")
                for ref in refs:
                    st.markdown(f"- {ref}")

    # --- Export ---
    st.markdown("---")
    st.markdown("### Export")
    ex1, ex2 = st.columns(2)
    with ex1:
        if spec_results:
            csv_df = pd.DataFrame([{
                "Parameter": s.get("parameter", ""),
                "SOP Criteria": s.get("sop_criteria", ""),
                "OOS Result": s.get("oos_result") or "",
                "Status": s.get("status", ""),
                "Deviation": s.get("deviation") or "",
            } for s in spec_results])
            st.download_button(
                "Export Specifications (CSV)",
                csv_df.to_csv(index=False),
                f"sop_oos_specs_{datetime.now():%Y%m%d_%H%M}.csv",
                "text/csv",
                use_container_width=True,
            )
    with ex2:
        st.download_button(
            "Export Full Report (JSON)",
            json.dumps(r, indent=2, default=str),
            f"sop_oos_report_{datetime.now():%Y%m%d_%H%M}.json",
            "application/json",
            use_container_width=True,
        )


# =============================================================================
# HISTORY TAB
# =============================================================================

def section_history():
    st.markdown('<div class="section-header">Analysis History</div>', unsafe_allow_html=True)

    records = get_history()
    if not records:
        st.info("No analysis history yet. Upload SOP and OOS documents to get started.")
        return

    st.caption("Local history is stored on this machine and remains visible even when the API server is offline.")

    for rec in records:
        ts = rec.get("timestamp", "")
        sop = rec.get("sop_name", "?")
        oos = rec.get("oos_name", "?")
        rate = rec.get("compliance_rate", 0)
        disp = rec.get("batch_disposition", "?")
        nc = rec.get("non_compliant", 0)

        rate_color = COLORS["success"] if rate >= 80 else (COLORS["warning"] if rate >= 50 else COLORS["danger"])
        disp_color = COLORS["danger"] if disp == "REJECTED" else COLORS["success"]

        col1, col2, col3, col4, col5 = st.columns([3.4, 1.7, 1, 1, 1])
        with col1:
            st.markdown(f"**{sop}** vs **{oos}**")
            detail = f"{ts[:19]} | {nc} non-compliant parameter(s)"
            st.markdown(f'<div class="history-meta">{detail}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span style="color: {disp_color}; font-weight: bold;">{disp}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span style="color: {rate_color}; font-weight: bold;">{rate}%</span>', unsafe_allow_html=True)
        with col4:
            if st.button("View", key=f"view_{rec['id']}"):
                full = get_history_record(rec["id"])
                if full and full.get("result_json"):
                    st.session_state.results = json.loads(full["result_json"])
                    st.rerun()
        with col5:
            if st.button("Delete", key=f"del_{rec['id']}"):
                delete_history_record(rec["id"])
                st.rerun()
        st.markdown("---")


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_page()
    render_sidebar()
    render_header()
    server_available = api_ok()

    tab_analyze, tab_history = st.tabs(["Analyze", "History"])

    with tab_analyze:
        if not server_available:
            show_connection_error()
        elif st.session_state.results:
            section_results()
        else:
            section_upload_and_analyze()

    with tab_history:
        section_history()


if __name__ == "__main__":
    main()
