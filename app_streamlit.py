#!/usr/bin/env python3

import time
import requests
from datetime import datetime
from typing import Optional, List, Dict

import streamlit as st


API_URL = "http://localhost:8000"
__ORANGE = "#D04A02"
__DARK = "#2D2D2D"

# Default KPIs for analysis
DEFAULT_KPIS = [
    "STRATIFICATION FACTORS",
    "RANDOMIZATION VISIT",
    "INCLUSION CRITERIA",
    "EXCLUSION CRITERIA",
    "DOSING SCHEDULE",
    "VISIT SCHEDULE",
]


def api_get(endpoint: str, timeout: int = 5) -> Optional[dict]:
    """GET request to API"""
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(endpoint: str, **kwargs) -> Optional[dict]:
    """POST request to API"""
    try:
        r = requests.post(f"{API_URL}{endpoint}", timeout=kwargs.pop("timeout", 60), **kwargs)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_delete(endpoint: str) -> bool:
    """DELETE request to API"""
    try:
        return requests.delete(f"{API_URL}{endpoint}", timeout=5).ok
    except Exception:
        return False


def api_ok() -> bool:
    """Check API health"""
    return api_get("/") is not None


def get_protocols() -> List[dict]:
    """Get all protocols"""
    result = api_get("/protocols")
    return result.get("protocols", []) if result else []


def get_system_info() -> dict:
    """Get system info"""
    return api_get("/system/info") or {}


def upload_protocol(file) -> Optional[dict]:
    """Upload protocol file"""
    return api_post("/protocols", files={"file": (file.name, file.getvalue())})


def submit_analysis(protocol_id: str, requirements_file, kpis: List[str]) -> Optional[dict]:
    """Submit KPI analysis job"""
    return api_post(
        "/analyze/kpis",
        files={"requirements": (requirements_file.name, requirements_file.getvalue())},
        data={"protocol_id": protocol_id, "kpis": ",".join(kpis)}
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


def setup_page():
    """Configure page and styling"""
    st.set_page_config(page_title="_ Compliance Analyzer", page_icon="📊", layout="wide")

    st.markdown(f"""
    <style>
        .main .block-container {{ max-width: 1000px; padding-top: 1rem; }}
        .header {{ background: {__DARK}; padding: 1rem 1.5rem; border-radius: 10px;
                   margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; }}
        .logo {{ color: {__ORANGE}; font-size: 2rem; font-weight: bold; }}
        .title {{ color: white; font-size: 1.3rem; }}
        .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1.5rem; margin: 0.5rem 0; }}
        .step {{ display: flex; align-items: center; gap: 10px; background: #f9fafb;
                 padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; }}
        .step-num {{ background: {__ORANGE}; color: white; width: 28px; height: 28px;
                     border-radius: 50%; display: flex; align-items: center; justify-content: center;
                     font-weight: bold; font-size: 0.85rem; }}
        .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px;
                  font-size: 0.8rem; font-weight: 500; }}
        .badge-ok {{ background: #d1fae5; color: #065f46; }}
        .badge-warn {{ background: #fef3c7; color: #92400e; }}
        .badge-err {{ background: #fee2e2; color: #991b1b; }}
        .kpi {{ display: inline-block; background: {__ORANGE}; color: white;
                padding: 0.2rem 0.5rem; border-radius: 10px; margin: 2px; font-size: 0.8rem; }}
        .stButton > button {{ background: {__ORANGE}; color: white; border: none; border-radius: 6px; }}
        .stButton > button:hover {{ background: #b03d00; }}
        #MainMenu, footer {{ visibility: hidden; }}
    </style>
    <div class="header">
        <div class="logo">_</div>
        <div class="title">Clinical Trial Compliance Analyzer</div>
    </div>
    """, unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.session_state.results = None


def page_home():
    """Home/Dashboard page"""
    st.markdown("## Welcome")

    protocols = get_protocols()
    info = get_system_info()

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Protocols", len(protocols))
    col2.metric("AI Agent", "Active" if info.get("agent_mode") else "Basic")
    col3.metric("Status", "Online" if api_ok() else "Offline")

    st.markdown("---")
    st.markdown("### Quick Start")

    # Steps
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="step"><div class="step-num">1</div><b>Upload Protocol</b></div>
            <p style="color: #6b7280; font-size: 0.9rem;">Add your clinical trial protocol (PDF/TXT)</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="step"><div class="step-num">2</div><b>Enter KPIs</b></div>
            <p style="color: #6b7280; font-size: 0.9rem;">Define KPIs and upload requirements</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <div class="step"><div class="step-num">3</div><b>Get Results</b></div>
            <p style="color: #6b7280; font-size: 0.9rem;">AI analyzes and provides findings</p>
        </div>
        """, unsafe_allow_html=True)

    # Status
    if protocols:
        st.success(f"✓ {len(protocols)} protocol(s) ready. Go to **Analyze** tab!")
    else:
        st.info("📁 Upload a protocol in **Protocols** tab to start.")


def page_protocols():
    """Protocols management page"""
    st.markdown("## Protocols")

    # Upload section
    file = st.file_uploader("Upload protocol", type=["pdf", "txt", "md"])
    if file:
        col1, col2 = st.columns([4, 1])
        col1.info(f"📄 {file.name} ({file.size/1024:.1f} KB)")
        if col2.button("Upload", type="primary"):
            with st.spinner("Uploading..."):
                if upload_protocol(file):
                    st.success("✓ Uploaded!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Upload failed")

    st.markdown("---")
    st.markdown("### Your Protocols")

    # List protocols
    protocols = get_protocols()
    if not protocols:
        st.caption("No protocols yet. Upload one above.")
    else:
        for p in protocols:
            col1, col2, col3 = st.columns([5, 2, 1])
            col1.markdown(f"**{p['name']}**")
            col2.caption(f"{p['file_size']/1024:.1f} KB • {p['indexed_at'][:10]}")
            if col3.button("🗑", key=f"del_{p['protocol_id']}"):
                api_delete(f"/protocols/{p['protocol_id']}")
                st.rerun()


def page_analyze():
    """Analysis page"""
    st.markdown("## Analyze")

    # Show results if available
    if st.session_state.results:
        show_results()
        return

    protocols = get_protocols()
    if not protocols:
        st.warning("⚠️ Upload a protocol first in **Protocols** tab.")
        return

    # Step 1: Select Protocol
    st.markdown('<div class="step"><div class="step-num">1</div><b>Select Protocol</b></div>', unsafe_allow_html=True)
    protocol_map = {p["name"]: p["protocol_id"] for p in protocols}
    selected = st.selectbox("Protocol", list(protocol_map.keys()), label_visibility="collapsed")
    protocol_id = protocol_map[selected]

    # Step 2: Upload Requirements
    st.markdown('<div class="step"><div class="step-num">2</div><b>Upload Requirements</b></div>', unsafe_allow_html=True)
    requirements = st.file_uploader("Requirements", type=["pdf", "txt", "md"], label_visibility="collapsed")
    if requirements:
        st.success(f"✓ {requirements.name}")

    # Step 3: Enter KPIs
    st.markdown('<div class="step"><div class="step-num">3</div><b>Enter KPIs</b></div>', unsafe_allow_html=True)
    st.caption("Default KPIs are pre-filled. Add more or modify as needed (one per line):")
    kpi_text = st.text_area(
        "KPIs",
        value="\n".join(DEFAULT_KPIS),
        height=150,
        label_visibility="collapsed"
    )
    kpis = [k.strip() for k in kpi_text.split("\n") if k.strip()]

    if kpis:
        st.markdown("".join([f'<span class="kpi">{k}</span>' for k in kpis]), unsafe_allow_html=True)

    st.markdown("---")

    # Submit
    ready = protocol_id and requirements and kpis
    if st.button("🚀 Start Analysis", type="primary", disabled=not ready):
        run_analysis(protocol_id, requirements, kpis)


def run_analysis(protocol_id: str, requirements_file, kpis: List[str]):
    """Run analysis and poll for results"""
    job = submit_analysis(protocol_id, requirements_file, kpis)
    if not job:
        st.error("Failed to start analysis")
        return

    job_id = job["job_id"]
    progress = st.progress(0.0)
    status = st.empty()
    status.info("⏳ Analyzing...")

    start = time.time()
    while time.time() - start < 300:
        j = get_job(job_id)
        if j:
            if j["status"] == "completed":
                progress.progress(1.0)
                status.success("✓ Complete!")
                result = get_job_result(job_id)
                if result and result.get("result"):
                    st.session_state.results = result["result"]
                    time.sleep(0.5)
                    st.rerun()
                return
            elif j["status"] == "failed":
                status.error(f"Failed: {j.get('error', 'Unknown')}")
                return
            else:
                progress.progress(min((time.time() - start) / 300 * 0.9, 0.9))
        time.sleep(2)

    status.warning("Taking longer than expected. Check History.")


def show_results():
    """Display analysis results"""
    r = st.session_state.results
    summary = r.get("summary", {})

    # Header
    col1, col2 = st.columns([3, 1])
    col1.markdown("### Results")
    if col2.button("🔄 New Analysis"):
        st.session_state.results = None
        st.rerun()

    # Metrics
    rate = summary.get("compliance_rate", 0)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Compliance", f"{rate:.0f}%")
    col2.metric("Total", summary.get("total_requirements", 0))
    col3.metric("Followed", summary.get("followed", 0))
    col4.metric("Partial", summary.get("partial", 0))
    col5.metric("Not Met", summary.get("not_followed", 0))

    # Alert
    if summary.get("not_followed", 0) > 0:
        st.error(f"⚠️ {summary['not_followed']} KPI(s) need attention!")
    elif summary.get("partial", 0) > 0:
        st.warning(f"⚡ {summary['partial']} KPI(s) partially met")
    else:
        st.success("✓ All compliant!")

    st.markdown("---")

    # Download report
    report = generate_report(r)
    if report:
        st.download_button("📥 Download Report", report, f"report_{datetime.now():%Y%m%d}.md", "text/markdown")

    st.markdown("---")
    st.markdown("### Details")

    # Detailed results
    for item in r.get("detailed_results", []):
        kpi = item.get("kpi", item.get("requirement", "?"))
        status = item.get("compliance_status", "?")
        conf = item.get("confidence_score", 0) * 100

        badge_cls = {"followed": "badge-ok", "partial": "badge-warn", "not_followed": "badge-err"}.get(status, "")
        badge_txt = {"followed": "✓ Followed", "partial": "~ Partial", "not_followed": "✗ Not Met"}.get(status, status)

        with st.expander(f"{kpi[:50]}{'...' if len(kpi) > 50 else ''}", expanded=(status == "not_followed")):
            st.markdown(f'<span class="badge {badge_cls}">{badge_txt}</span> &nbsp; Confidence: {conf:.0f}%', unsafe_allow_html=True)

            # Explanation
            exp = item.get("explanation", "")
            if exp:
                st.markdown(exp[:400] + ("..." if len(exp) > 400 else ""))

            # Gaps
            gaps = item.get("gaps_identified", "")
            if gaps:
                st.markdown(f"**Gaps:** {gaps[:300]}")

            # Steps detail
            s1 = item.get("step_3_1_conditions", {})
            s2 = item.get("step_3_2_checks", [])
            s3 = item.get("step_3_3_judgement", {})

            if s1 or s2 or s3:
                t1, t2, t3 = st.tabs(["Conditions (Protocol)", "Evidence (Requirements)", "Judgement"])

                with t1:
                    conditions = s1.get("conditions", [])
                    sources = s1.get("source_sections", [])
                    quotes = s1.get("source_quotes", [])
                    for i, c in enumerate(conditions[:5]):
                        src = sources[i] if i < len(sources) else "N/A"
                        quote = quotes[i] if i < len(quotes) else ""
                        st.markdown(f"**{i+1}. {c[:100]}**")
                        st.caption(f"📍 Source: {src}")
                        if quote:
                            st.caption(f'📝 "{quote[:150]}..."')
                        st.markdown("---")

                with t2:
                    for c in s2[:5]:
                        lvl = c.get("coverage_level", "none")
                        icon = {"full": "✅", "partial": "⚠️", "none": "❌"}.get(lvl, "•")
                        st.markdown(f"{icon} **{c.get('condition', '')[:80]}**")
                        st.caption(f"📍 Evidence: {c.get('evidence_source', 'N/A')}")
                        if c.get("evidence_quote"):
                            st.caption(f'📝 "{c.get("evidence_quote", "")[:150]}..."')
                        if lvl != "full" and c.get("gap_description"):
                            st.warning(f"Gap: {c.get('gap_description', '')[:150]}")
                        st.markdown("---")

                with t3:
                    st.markdown("**Reasoning:**")
                    st.markdown(s3.get("reasoning", exp)[:400])
                    if s3.get("evidence_summary"):
                        st.markdown("**Evidence Summary:**")
                        st.markdown(s3.get("evidence_summary", "")[:300])
                    if s3.get("gaps_identified"):
                        st.markdown("**Gaps:**")
                        st.warning(s3.get("gaps_identified", "")[:300])


def page_history():
    """Job history page"""
    st.markdown("## History")

    jobs_data = api_get("/jobs")
    jobs = jobs_data.get("jobs", []) if jobs_data else []

    if not jobs:
        st.caption("No history yet.")
        return

    for j in sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True):
        status = j.get("status", "?")
        icon = {"completed": "🟢", "processing": "🟡", "pending": "🔵", "failed": "🔴"}.get(status, "⚪")

        col1, col2, col3 = st.columns([4, 2, 1])
        col1.markdown(f"**{j.get('job_type', 'Analysis')}** • {j['job_id'][:8]}...")
        col2.markdown(f"{icon} {status.title()}")

        if status == "completed":
            if col3.button("View", key=f"view_{j['job_id']}"):
                result = get_job_result(j["job_id"])
                if result and result.get("result"):
                    st.session_state.results = result["result"]
                    st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    setup_page()

    if not api_ok():
        st.error("⚠️ Cannot connect to API. Run: `python server.py`")
        return

    tabs = st.tabs(["🏠 Home", "📁 Protocols", "🔍 Analyze", "📋 History"])

    with tabs[0]:
        page_home()
    with tabs[1]:
        page_protocols()
    with tabs[2]:
        page_analyze()
    with tabs[3]:
        page_history()


if __name__ == "__main__":
    main()
