#!/usr/bin/env python3
"""
Clinical Trial Compliance Analyzer - Streamlit UI
Sidebar layout with document upload and analysis
"""

import time
import hashlib
import json
import sqlite3
import os
import requests
import html
import textwrap
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict

import streamlit as st


API_URL = "http://localhost:8000"
HISTORY_DB_PATH = "data/analysis_history.db"



def init_history_db():
    """Initialize the SQLite database for storing analysis history"""
    os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            protocol_name TEXT,
            requirements_name TEXT,
            topics_analyzed TEXT,
            total_topics INTEGER,
            matches INTEGER,
            partial_matches INTEGER,
            no_matches INTEGER,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_to_history(
    protocol_name: str,
    requirements_name: str,
    topics: List[str],
    results: dict
) -> int:
    """Save analysis results to history database. Returns the new record ID."""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()

    summary = results.get("summary", {})

    cursor.execute("""
        INSERT INTO analysis_history (
            timestamp, protocol_name, requirements_name, topics_analyzed,
            total_topics, matches, partial_matches, no_matches, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        protocol_name,
        requirements_name,
        json.dumps(topics),
        summary.get("total_requirements", 0),
        summary.get("followed", 0),
        summary.get("partial", 0),
        summary.get("not_followed", 0),
        json.dumps(results)
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_history(limit: int = 50) -> List[dict]:
    """Get analysis history records"""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM analysis_history
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_history_record(record_id: int) -> Optional[dict]:
    """Get a single history record by ID"""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_history_record(record_id: int) -> bool:
    """Delete a history record"""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted



@st.cache_resource
def get_http_session() -> requests.Session:
    """Create a cached HTTP session with connection pooling and retries"""
    session = requests.Session()

    # Retry strategy for transient errors
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )

    # Mount adapter with connection pooling
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=retry_strategy
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# Color scheme - PwC Brand Colors
COLORS = {
    "primary": "#D04A02",      # PwC Orange
    "secondary": "#515350",    # Black
    "dark": "#2B2B2B",         # Dark Gray
    "success": "#00854D",      # Dark Green
    "warning": "#FFA500",      # Orange Warning
    "danger": "#DC291E",       # Red
    "light_bg": "#515350",     # Light Gray Background
    "border": "#DDDDDD",       # Border Gray
    "text_secondary": "#666666", # Secondary Text
}

# Default KPIs for analysis
DEFAULT_KPIS = [
    "STRATIFICATION FACTORS",
    "RANDOMIZATION VISIT",
    "INCLUSION CRITERIA",
    "EXCLUSION CRITERIA",
    "DOSING SCHEDULE",
    "VISIT SCHEDULE",
]

# Common KPIs that can be added
COMMON_KPIS = [
    "PRIMARY ENDPOINTS",
    "SECONDARY ENDPOINTS",
    "SAFETY ASSESSMENTS",
    "STUDY DURATION",
    "SAMPLE SIZE",
    "WASHOUT PERIOD",
    "CONCOMITANT MEDICATIONS",
    "DISCONTINUATION CRITERIA",
]



def api_get(endpoint: str, timeout: int = 5) -> Optional[dict]:
    """GET request to API with connection pooling"""
    try:
        session = get_http_session()
        r = session.get(f"{API_URL}{endpoint}", timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(endpoint: str, **kwargs) -> Optional[dict]:
    """POST request to API with connection pooling"""
    try:
        session = get_http_session()
        r = session.post(f"{API_URL}{endpoint}", timeout=kwargs.pop("timeout", 60), **kwargs)
        if r.ok:
            return r.json()
        else:
            # Log error details for debugging
            try:
                error_detail = r.json().get("detail", r.text)
            except Exception:
                error_detail = r.text
            print(f"API Error {r.status_code}: {error_detail}")
            return None
    except Exception as e:
        print(f"Request error: {e}")
        return None


def api_delete(endpoint: str) -> bool:
    """DELETE request to API with connection pooling"""
    try:
        session = get_http_session()
        return session.delete(f"{API_URL}{endpoint}", timeout=5).ok
    except Exception:
        return False


def api_ok() -> bool:
    """Check API health"""
    return api_get("/") is not None


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_protocols() -> List[dict]:
    """Get all protocols (cached)"""
    result = api_get("/protocols")
    return result.get("protocols", []) if result else []


def clear_protocols_cache():
    """Clear the protocols cache"""
    get_protocols.clear()


def get_system_info() -> dict:
    """Get system info"""
    return api_get("/system/info") or {}


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()


def upload_protocol(file) -> Optional[dict]:
    """
    Upload protocol file with hash-based deduplication.

    Flow: Upload → Hash Check → Exists?
          ├─ Yes: Return existing (O(1) lookup)
          └─ No:  Save → Chunk → Embed → Register hash
    """
    result = api_post("/protocols", files={"file": (file.name, file.getvalue())})
    if result:
        clear_protocols_cache()  # Invalidate cache after upload
    return result


def check_protocol_exists(file_hash: str) -> Optional[dict]:
    """Check if protocol with given hash already exists"""
    result = api_get(f"/protocols/hash/{file_hash}")
    return result if result else None


def submit_analysis(protocol_id: str, requirements_file, kpis: List[str], concurrent: bool = False) -> Optional[dict]:
    """Submit KPI analysis job"""
    return api_post(
        "/analyze/kpis",
        files={"requirements": (requirements_file.name, requirements_file.getvalue())},
        data={
            "protocol_id": protocol_id,
            "kpis": ",".join(kpis),
            "concurrent": str(concurrent).lower()
        }
    )


def get_job(job_id: str) -> Optional[dict]:
    """Get job status"""
    return api_get(f"/jobs/{job_id}")


def get_job_result(job_id: str) -> Optional[dict]:
    """Get job result"""
    return api_get(f"/jobs/{job_id}/result")


def generate_report(data: dict) -> Optional[str]:
    """Generate markdown report"""
    result = api_post("/report/generate", json=data, timeout=30)
    return result.get("markdown") if result else None


# =============================================================================
# PAGE SETUP & STYLING
# =============================================================================

def setup_page():
    """Configure page and styling"""
    # Initialize history database
    init_history_db()

    st.set_page_config(
        page_title="Clinical Trial Document Checker",
        page_icon="",
        layout="wide"
    )

    st.markdown(f"""
    <style>
        /* Main container */
        .main .block-container {{
            max-width: 1400px;
            padding-top: 1rem;
            background-color: {COLORS['light_bg']};
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, {COLORS['light_bg']} 100%);
            border-right: 1px solid {COLORS['border']};
        }}
        [data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
        }}

        /* Section styling - PwC */
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
        .section-divider {{
            border: none;
            border-top: 2px solid {COLORS['border']};
            margin: 2rem 0;
        }}

        /* App header - PwC Style */
        .app-header {{
            background: {COLORS['secondary']};
            padding: 2rem 2.5rem;
            border-radius: 0;
            margin-bottom: 2rem;
            border-left: 6px solid {COLORS['primary']};
        }}
        .app-header-icon {{
            font-size: 2rem;
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

        /* Sidebar sections - PwC Style */
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
        .sidebar-section ul {{
            color: {COLORS['text_secondary']};
            font-size: 0.9rem;
            margin: 0.5rem 0;
            padding-left: 1.2rem;
        }}
        .sidebar-section li {{
            margin: 0.5rem 0;
        }}

        /* Feature checkmarks */
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

        /* How it works steps */
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

        /* Upload section */
        .upload-label {{
            font-weight: 500;
            color: {COLORS['dark']};
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }}
        .upload-hint {{
            color: #9ca3af;
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }}

        /* File info display */
        .file-info {{
            background: #f0fdf4;
            border: 1px solid {COLORS['success']};
            border-radius: 6px;
            padding: 0.6rem 0.8rem;
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }}
        .file-info-success {{
            color: {COLORS['success']};
        }}
        .file-info-name {{
            color: {COLORS['dark']};
            font-weight: 500;
        }}
        .file-info-reused {{
            background: #fef3c7;
            border-color: {COLORS['warning']};
        }}
        .file-info-reused .file-info-success {{
            color: {COLORS['warning']};
        }}

        /* KPI tags */
        .kpi-tag {{
            display: inline-block;
            background: {COLORS['primary']};
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 10px;
            margin: 2px;
            font-size: 0.75rem;
        }}

        /* Summary cards - PwC Style */
        .summary-card {{
            background: white;
            border: 1px solid {COLORS['border']};
            border-radius: 0;
            padding: 1.75rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
            transition: all 0.2s;
        }}
        .summary-card:hover {{
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
        }}
        .summary-card-score {{
            border-top: 4px solid {COLORS['primary']};
        }}
        .summary-card-followed {{
            border-top: 4px solid {COLORS['success']};
        }}
        .summary-card-partial {{
            border-top: 4px solid {COLORS['warning']};
        }}
        .summary-card-failed {{
            border-top: 4px solid {COLORS['danger']};
        }}
        .summary-value {{
            font-size: 3rem;
            font-weight: 300;
            margin-bottom: 0.5rem;
            line-height: 1;
        }}
        .summary-value-score {{ color: {COLORS['primary']}; }}
        .summary-value-followed {{ color: {COLORS['success']}; }}
        .summary-value-partial {{ color: {COLORS['warning']}; }}
        .summary-value-failed {{ color: {COLORS['danger']}; }}
        .summary-label {{
            color: {COLORS['secondary']};
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        /* Status badges */
        .status-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .status-followed {{
            background: #d1fae5;
            color: #065f46;
        }}
        .status-partial {{
            background: #fef3c7;
            color: #92400e;
        }}
        .status-not_followed {{
            background: #fee2e2;
            color: #991b1b;
        }}

        /* Compare button - PwC Style */
        .compare-btn {{
            background: {COLORS['primary']} !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 1rem 2rem !important;
            border-radius: 0 !important;
            border: none !important;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: all 0.2s;
        }}
        .compare-btn:hover {{
            background: #B03F02 !important;
        }}

        /* Progress styling */
        .progress-kpi {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.3rem 0;
            color: #6b7280;
            font-size: 0.85rem;
        }}
        .progress-kpi-active {{
            color: {COLORS['primary']};
            font-weight: 500;
        }}
        .progress-kpi-done {{
            color: {COLORS['success']};
        }}

        /* Button styling */
        .stButton > button {{
            border-radius: 6px;
            font-weight: 500;
        }}

        /* Hide Streamlit elements */
        #MainMenu, footer {{ visibility: hidden; }}

        /* Alert styling */
        .alert-success {{
            background: #f0fdf4;
            border-left: 4px solid {COLORS['success']};
            padding: 0.75rem 1rem;
            border-radius: 6px;
            color: #166534;
            font-size: 0.9rem;
        }}

        /* Tab styling */
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

        /* History table styling */
        .history-record {{
            background: white;
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "protocol_id" not in st.session_state:
        st.session_state.protocol_id = None
    if "protocol_name" not in st.session_state:
        st.session_state.protocol_name = None
    if "protocol_reused" not in st.session_state:
        st.session_state.protocol_reused = False
    if "requirements_file" not in st.session_state:
        st.session_state.requirements_file = None
    if "selected_kpis" not in st.session_state:
        st.session_state.selected_kpis = []
    if "results" not in st.session_state:
        st.session_state.results = None
    if "job_id" not in st.session_state:
        st.session_state.job_id = None
    if "filter_status" not in st.session_state:
        st.session_state.filter_status = "all"
    if "concurrent_mode" not in st.session_state:
        st.session_state.concurrent_mode = False
    if "viewing_history" not in st.session_state:
        st.session_state.viewing_history = False
    if "history_record_id" not in st.session_state:
        st.session_state.history_record_id = None


def render_sidebar():
    """Render the sidebar with app info"""
    with st.sidebar:
        # PwC Branding section
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0 1.5rem 0; border-bottom: 2px solid {COLORS['border']};">
            <div style="font-size: 2rem; font-weight: 700; color: {COLORS['primary']};">PwC</div>
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-top: 0.25rem;">Life Sciences & Healthcare</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # About section - simplified
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">What This Does</div>
            <p>
                Compares your clinical trial documents to check if they match.
                Finds differences and shows what needs attention.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Key Features - simplified
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Features</div>
            <div class="feature-item"><span class="feature-check">&check;</span> Works with PDF, TXT, MD files</div>
            <div class="feature-item"><span class="feature-check">&check;</span> AI reads and compares documents</div>
            <div class="feature-item"><span class="feature-check">&check;</span> Shows what matches and what doesn't</div>
            <div class="feature-item"><span class="feature-check">&check;</span> Downloadable reports</div>
        </div>
        """, unsafe_allow_html=True)

        # How It Works - simplified
        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">How To Use</div>
            <div class="step-item"><div class="step-num">1</div> Upload your two documents</div>
            <div class="step-item"><div class="step-num">2</div> Pick topics to compare</div>
            <div class="step-item"><div class="step-num">3</div> Click Compare</div>
            <div class="step-item"><div class="step-num">4</div> Review results</div>
        </div>
        """, unsafe_allow_html=True)



def render_header():
    """Render the main content header"""
    st.markdown(f"""
    <div class="app-header">
        <div style="display: flex; align-items: center; gap: 1.5rem;">
            <div>
                <div class="app-header-title">Clinical Trial Document Checker</div>
                <div class="app-header-subtitle">Compare your protocol and requirements to find differences</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_connection_error():
    """Show API connection error"""
    st.error("Cannot connect to the server.")
    st.markdown("""
    **The server needs to be running first.**

    Open a terminal and run:
    ```
    python server.py
    ```
    Then refresh this page.
    """)


# =============================================================================
# DOCUMENT UPLOAD SECTION
# =============================================================================

def section_upload_and_analyze():
    """Render document upload and analysis section"""
    st.markdown('<div class="section-header">Upload & Compare</div>', unsafe_allow_html=True)

    # Success message placeholder
    if st.session_state.get("upload_success"):
        st.markdown(f'<div class="alert-success">&check; {st.session_state.upload_success}</div>', unsafe_allow_html=True)
        st.session_state.upload_success = None

    st.markdown("#### Step 1: Upload Your Documents")

    col1, col2 = st.columns(2)

    # Protocol File Upload
    with col1:
        st.markdown('<div class="upload-label">Protocol Document</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-hint">Your official trial protocol</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-hint">PDF, TXT, or MD (max 200MB)</div>', unsafe_allow_html=True)

        protocol_file = st.file_uploader(
            "Protocol",
            type=["pdf", "txt", "md"],
            key="protocol_upload",
            label_visibility="collapsed"
        )

        if protocol_file:
            # Compute hash for deduplication
            file_content = protocol_file.getvalue()
            file_hash = compute_file_hash(file_content)

            # Check if we need to process this file
            if st.session_state.protocol_id is None or st.session_state.get("last_protocol_hash") != file_hash:
                with st.spinner("Processing protocol..."):
                    # Upload to server (server handles hash check)
                    try:
                        result = upload_protocol(protocol_file)

                        if result:
                            st.session_state.protocol_id = result.get("protocol_id")
                            st.session_state.protocol_name = protocol_file.name
                            st.session_state.protocol_reused = result.get("reused", False)
                            st.session_state.last_protocol_hash = file_hash
                        else:
                            st.error("Could not upload file. Is the server running?")
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")

            # Show file info
            if st.session_state.protocol_id:
                reused_class = "file-info-reused" if st.session_state.protocol_reused else ""
                reused_text = " (cached)" if st.session_state.protocol_reused else ""
                st.markdown(f"""
                <div class="file-info {reused_class}">
                    <span class="file-info-success">&check;</span>
                    <span class="file-info-name">{st.session_state.protocol_name}{reused_text}</span>
                </div>
                """, unsafe_allow_html=True)

    # Requirements File Upload
    with col2:
        st.markdown('<div class="upload-label">Requirements Document</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-hint">Document to check against the protocol</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-hint">PDF, TXT, or MD (max 200MB)</div>', unsafe_allow_html=True)

        requirements_file = st.file_uploader(
            "Requirements",
            type=["pdf", "txt", "md"],
            key="requirements_upload",
            label_visibility="collapsed"
        )

        if requirements_file:
            st.session_state.requirements_file = requirements_file
            st.markdown(f"""
            <div class="file-info">
                <span class="file-info-success">&check;</span>
                <span class="file-info-name">{requirements_file.name}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Topic Selection
    st.markdown("#### Step 2: Pick Topics to Compare")

    # Multi-select for KPIs
    all_kpis = DEFAULT_KPIS + COMMON_KPIS
    selected_kpis = st.multiselect(
        "Select which topics to check",
        options=all_kpis,
        default=st.session_state.selected_kpis if st.session_state.selected_kpis else ["STRATIFICATION FACTORS"],
        key="kpi_multiselect",
        label_visibility="collapsed",
        help="Choose one or more topics. The AI will compare these sections between your documents."
    )
    st.session_state.selected_kpis = selected_kpis

    # Show selected KPIs as tags
    if selected_kpis:
        kpi_tags = "".join([f'<span class="kpi-tag">{k}</span>' for k in selected_kpis])
        st.markdown(kpi_tags, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Processing options
    col1, col2 = st.columns([1, 2])
    with col1:
        concurrent_mode = st.toggle(
            "Fast Mode",
            value=st.session_state.get("concurrent_mode", False),
            help="Analyzes all topics at once (faster)"
        )
        st.session_state.concurrent_mode = concurrent_mode

    if concurrent_mode:
        st.caption("Analyzing all topics at once")
    else:
        st.caption("Analyzing topics one by one")

    st.markdown("<br>", unsafe_allow_html=True)

    # Compare Button
    protocol_ready = st.session_state.protocol_id is not None
    requirements_ready = st.session_state.requirements_file is not None
    kpis_ready = len(selected_kpis) > 0

    compare_disabled = not (protocol_ready and requirements_ready and kpis_ready)

    st.markdown("#### Step 3: Run Comparison")
    if st.button("Compare Documents", type="primary", disabled=compare_disabled, use_container_width=True):
        run_analysis(st.session_state.protocol_id, st.session_state.requirements_file, selected_kpis, concurrent_mode)


# =============================================================================
# ANALYSIS EXECUTION
# =============================================================================

def run_analysis(protocol_id: str, requirements_file, kpis: List[str], concurrent: bool = False):
    """Run analysis with progress display"""
    job = submit_analysis(protocol_id, requirements_file, kpis, concurrent)
    if not job:
        st.error("Could not start. Make sure the server is running.")
        return

    if concurrent:
        st.info("Analyzing all topics at once...")

    job_id = job["job_id"]
    st.session_state.job_id = job_id

    # Progress display
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    kpi_status = st.empty()

    # Show KPIs being processed
    kpi_html = ""
    for kpi in kpis:
        kpi_html += f'<div class="progress-kpi">⏳ {kpi}</div>'
    kpi_status.markdown(kpi_html, unsafe_allow_html=True)

    start = time.time()
    completed_count = 0

    while time.time() - start < 300:
        j = get_job(job_id)
        if j:
            if j["status"] == "completed":
                progress_bar.progress(1.0)
                status_text.success("Done!")

                # Update KPI status to all done
                kpi_html = ""
                for kpi in kpis:
                    kpi_html += f'<div class="progress-kpi progress-kpi-done">&check; {kpi}</div>'
                kpi_status.markdown(kpi_html, unsafe_allow_html=True)

                result = get_job_result(job_id)
                if result and result.get("result"):
                    st.session_state.results = result["result"]
                    # Save to history
                    try:
                        save_to_history(
                            protocol_name=st.session_state.get("protocol_name", "Unknown"),
                            requirements_name=requirements_file.name if requirements_file else "Unknown",
                            topics=kpis,
                            results=result["result"]
                        )
                    except Exception as e:
                        print(f"Failed to save to history: {e}")
                    time.sleep(0.5)
                    st.rerun()
                return

            elif j["status"] == "failed":
                status_text.error(f"Something went wrong: {j.get('error', 'Unknown error')}")
                return
            else:
                # Update progress
                elapsed = time.time() - start
                progress = min(elapsed / 180 * 0.9, 0.9)
                progress_bar.progress(progress)

                # Estimate which KPIs might be done
                estimated_done = int((elapsed / 180) * len(kpis))
                if estimated_done > completed_count:
                    completed_count = estimated_done
                    kpi_html = ""
                    for i, kpi in enumerate(kpis):
                        if i < completed_count:
                            kpi_html += f'<div class="progress-kpi progress-kpi-done">&check; {kpi}</div>'
                        elif i == completed_count:
                            kpi_html += f'<div class="progress-kpi progress-kpi-active">Checking: {kpi}</div>'
                        else:
                            kpi_html += f'<div class="progress-kpi">Waiting: {kpi}</div>'
                    kpi_status.markdown(kpi_html, unsafe_allow_html=True)

                status_text.info(f"Working... {int(progress * 100)}%")

        time.sleep(2)

    status_text.warning("Taking longer than usual. Please wait...")


# =============================================================================
# RESULTS SECTION
# =============================================================================

def render_executive_summary(summary: dict):
    """Render a simple executive summary banner"""
    followed = summary.get("followed", 0)
    partial = summary.get("partial", 0)
    not_followed = summary.get("not_followed", 0)
    total = summary.get("total_requirements", 0)

    # Single line executive summary
    if not_followed > 0:
        st.error(f"**{not_followed} of {total} topics need attention** — {followed} match, {partial} partial match")
    elif partial > 0:
        st.warning(f"**{partial} of {total} topics have minor differences** — {followed} match fully")
    else:
        st.success(f"**All {total} topics match** — No issues found")


def section_results():
    """Render results section"""
    r = st.session_state.results
    summary = r.get("summary", {})
    detailed_results = r.get("detailed_results", [])

    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

    # Header with actions
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        report = generate_report(r)
        if report:
            st.download_button(
                "Download Report",
                report,
                f"compliance_report_{datetime.now():%Y%m%d_%H%M}.md",
                "text/markdown",
                use_container_width=True
            )
    with col3:
        if st.button("Start Over", use_container_width=True):
            st.session_state.results = None
            st.session_state.filter_status = "all"
            st.rerun()

    # Executive summary banner
    render_executive_summary(summary)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Summary")

    # Build markdown table
    table_lines = [
        "| Topic | Protocol Summary | Requirement Summary | Confidence | Judgement |",
        "|-------|------------------|---------------------|------------|-----------|"  
    ]
    
    for item in detailed_results:
        kpi = item.get("kpi", item.get("requirement", "Unknown"))
        status = item.get("compliance_status", "unknown")
        
        # Status display
        status_map = {
            "followed": "Match",
            "partial": "Partial",
            "not_followed": "No Match"
        }
        status_display = status_map.get(status, "Unknown")
        
        s1 = item.get("step_3_1_conditions", {})
        s3 = item.get("step_3_3_judgement", {})
        
        # Protocol summary
        protocol_summary = s1.get("protocol_context", "—")
        if not protocol_summary or protocol_summary == "Error":
            conditions = s1.get("conditions", [])
            protocol_summary = "; ".join(conditions) if conditions else "—"
        
        # Requirement summary
        requirement_summary = s3.get("evidence_summary", "—")
        
        # Confidence
        confidence_score = item.get("confidence_score", 0.0)
        confidence_pct = f"{int(confidence_score * 100)}%"
        
        # Clean for markdown (remove pipes and newlines)
        kpi = kpi.replace("|", "\\|").replace("\n", " ")
        protocol_summary = protocol_summary.replace("|", "\\|").replace("\n", " ")
        requirement_summary = requirement_summary.replace("|", "\\|").replace("\n", " ")
        
        table_lines.append(
            f"| {kpi} | {protocol_summary} | {requirement_summary} | {confidence_pct} | {status_display} |"
        )
    
    st.markdown("\n".join(table_lines))

    # Export buttons
    col1, col2 = st.columns(2)
    with col1:
        if detailed_results:
            csv_data = []
            for item in detailed_results:
                status = item.get("compliance_status", "")
                s1 = item.get("step_3_1_conditions", {})
                s3 = item.get("step_3_3_judgement", {})
                
                csv_data.append({
                    "Topic": item.get("kpi", item.get("requirement", "")),
                    "Protocol Summary": s1.get("protocol_context", ""),
                    "Requirement Summary": s3.get("evidence_summary", ""),
                    "Confidence": item.get("confidence_score", 0.0),
                    "Judgement": {"followed": "Match", "partial": "Partial", "not_followed": "No Match"}.get(status, status)
                })

            csv_df = pd.DataFrame(csv_data)
            csv = csv_df.to_csv(index=False)

            st.download_button(
                "Export Summary (CSV)",
                csv,
                f"summary_{datetime.now():%Y%m%d_%H%M}.csv",
                "text/csv",
                use_container_width=True
            )

    section_detailed_analysis(detailed_results)


def section_detailed_analysis(detailed_results: List[dict]):
    """Render detailed analysis section with annotated data"""
    st.markdown("---")
    st.markdown("### Detailed Analysis")
    st.caption("Expand each topic to see the full comparison with source references")

    for item in detailed_results:
        kpi = item.get("kpi", item.get("requirement", "Unknown"))
        status = item.get("compliance_status", "unknown")

        # Simple status indicator without confidence score
        status_icon = {"followed": "[MATCH]", "partial": "[PARTIAL]", "not_followed": "[NO MATCH]"}.get(status, "[?]")

        with st.expander(f"{status_icon} {kpi}", expanded=False):
            s1 = item.get("step_3_1_conditions", {})
            s2 = item.get("step_3_2_checks", [])
            s3 = item.get("step_3_3_judgement", {})

            st.markdown("##### What the Protocol Says")
            conditions = s1.get("conditions", [])
            sources = s1.get("source_sections", [])

            if conditions:
                for i, condition in enumerate(conditions):
                    src = sources[i] if i < len(sources) else ""
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border-left: 3px solid {COLORS['primary']}; padding: 0.75rem 1rem; margin: 0.5rem 0;">
                        <div style="color: #333;">{condition}</div>
                        {f'<div style="color: #888; font-size: 0.8rem; margin-top: 0.25rem;">Source: {src}</div>' if src else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific requirements found in protocol for this topic.")

            st.markdown("")

            st.markdown("##### Evidence from Requirements Document")

            if s2:
                for check in s2:
                    coverage = check.get("coverage_level", "none")
                    condition_text = check.get("condition", "")
                    evidence = check.get("evidence_quote", "")
                    gap = check.get("gap_description", "")

                    # Color coding based on coverage
                    if coverage == "full":
                        border_color = COLORS['success']
                        label = "FOUND"
                    elif coverage == "partial":
                        border_color = COLORS['warning']
                        label = "PARTIAL"
                    else:
                        border_color = COLORS['danger']
                        label = "NOT FOUND"

                    st.markdown(f"""
                    <div style="border: 1px solid {border_color}; border-left: 4px solid {border_color}; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 4px;">
                        <div style="font-weight: 600; color: {border_color}; font-size: 0.75rem; margin-bottom: 0.25rem;">{label}</div>
                        <div style="color: #333; margin-bottom: 0.5rem;">{condition_text}</div>
                        {f'<div style="background: #f0f0f0; padding: 0.5rem; border-radius: 4px; font-style: italic; color: #555;">"{evidence}"</div>' if evidence else ''}
                        {f'<div style="color: {COLORS["danger"]}; margin-top: 0.5rem; font-size: 0.9rem;">Gap: {gap}</div>' if gap and coverage != "full" else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No evidence checks available.")

            st.markdown("")

            # ---------- CONCLUSION SECTION ----------
            st.markdown("##### Conclusion")
            reasoning = s3.get("reasoning", item.get("explanation", "")) if s3 else item.get("explanation", "")
            gaps = s3.get("gaps_identified", item.get("gaps_identified", "")) if s3 else item.get("gaps_identified", "")

            if reasoning:
                st.markdown(reasoning)

            if gaps and status != "followed":
                st.markdown(f"""
                <div style="background: #fff3cd; border-left: 4px solid {COLORS['warning']}; padding: 0.75rem 1rem; margin-top: 0.5rem;">
                    <div style="font-weight: 600; color: #856404; margin-bottom: 0.25rem;">Action Required</div>
                    <div style="color: #856404;">{gaps}</div>
                </div>
                """, unsafe_allow_html=True)
            elif status == "followed":
                st.success("No action needed - documents are aligned.")


def section_history():
    """Render history section showing past analyses"""
    st.markdown('<div class="section-header">Analysis History</div>', unsafe_allow_html=True)

    history = get_history(limit=50)

    if not history:
        st.info("No analysis history yet. Run your first analysis to see it here.")
        return

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Analyses", len(history))
    with col2:
        recent_matches = sum(1 for h in history[:10] if h["no_matches"] == 0)
        st.metric("Recent Success Rate", f"{recent_matches * 10}%")
    with col3:
        if history:
            last_date = datetime.fromisoformat(history[0]["timestamp"]).strftime("%b %d, %Y %H:%M")
            st.metric("Last Analysis", last_date)

    st.markdown("---")

    # History table
    st.markdown("### Past Analyses")

    history_table = []
    for record in history:
        timestamp = datetime.fromisoformat(record["timestamp"])
        topics = json.loads(record["topics_analyzed"]) if record["topics_analyzed"] else []

        # Status summary
        if record["no_matches"] > 0:
            status = f"Issues ({record['no_matches']})"
            status_color = "red"
        elif record["partial_matches"] > 0:
            status = f"Partial ({record['partial_matches']})"
            status_color = "orange"
        else:
            status = "All Match"
            status_color = "green"

        history_table.append({
            "ID": record["id"],
            "Date": timestamp.strftime("%Y-%m-%d %H:%M"),
            "Protocol": record["protocol_name"] or "—",
            "Requirements": record["requirements_name"] or "—",
            "Topics": len(topics),
            "Match": record["matches"],
            "Partial": record["partial_matches"],
            "No Match": record["no_matches"],
            "Status": status
        })

    df = pd.DataFrame(history_table)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Date": st.column_config.TextColumn("Date", width="medium"),
            "Protocol": st.column_config.TextColumn("Protocol", width="medium"),
            "Requirements": st.column_config.TextColumn("Requirements", width="medium"),
            "Topics": st.column_config.NumberColumn("Topics", width="small"),
            "Match": st.column_config.NumberColumn("Match", width="small"),
            "Partial": st.column_config.NumberColumn("Partial", width="small"),
            "No Match": st.column_config.NumberColumn("No Match", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small")
        }
    )

    st.markdown("---")

    # View details of a specific record
    st.markdown("### View Details")
    col1, col2 = st.columns([1, 3])

    with col1:
        record_ids = [r["id"] for r in history]
        selected_id = st.selectbox(
            "Select analysis ID",
            options=record_ids,
            format_func=lambda x: f"#{x}"
        )

    with col2:
        col_view, col_delete = st.columns(2)
        with col_view:
            if st.button("View Full Results", use_container_width=True):
                record = get_history_record(selected_id)
                if record:
                    result_data = json.loads(record["result_json"])
                    st.session_state.results = result_data
                    st.session_state.viewing_history = True
                    st.session_state.history_record_id = selected_id
                    st.rerun()
        with col_delete:
            if st.button("Delete", type="secondary", use_container_width=True):
                if delete_history_record(selected_id):
                    st.success(f"Deleted record #{selected_id}")
                    time.sleep(0.5)
                    st.rerun()

    # Show selected record summary
    if selected_id:
        record = get_history_record(selected_id)
        if record:
            st.markdown("---")
            timestamp = datetime.fromisoformat(record["timestamp"])
            topics = json.loads(record["topics_analyzed"]) if record["topics_analyzed"] else []

            st.markdown(f"""
            **Analysis #{record['id']}** — {timestamp.strftime("%B %d, %Y at %H:%M")}

            | | |
            |---|---|
            | **Protocol** | {record['protocol_name']} |
            | **Requirements** | {record['requirements_name']} |
            | **Topics Checked** | {', '.join(topics)} |
            | **Results** | {record['matches']} match, {record['partial_matches']} partial, {record['no_matches']} no match |
            """)


def main():
    """Main entry point"""
    setup_page()

    # Render sidebar
    render_sidebar()

    # Check API connection
    if not api_ok():
        render_header()
        show_connection_error()
        return

    render_header()

    # Tab navigation
    tab1, tab2 = st.tabs(["New Analysis", "History"])

    with tab1:
        # Show results or upload section
        if st.session_state.results:
            # Add "Back to History" button if viewing from history
            if st.session_state.get("viewing_history"):
                if st.button("Back to History"):
                    st.session_state.results = None
                    st.session_state.viewing_history = False
                    st.rerun()
            section_results()
        else:
            section_upload_and_analyze()

    with tab2:
        section_history()


if __name__ == "__main__":
    main()