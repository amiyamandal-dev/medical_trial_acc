#!/usr/bin/env python3
"""
Streamlit Web Interface for Clinical Trial Protocol Compliance Analysis

This application provides an interactive web interface for:
1. Basic vector similarity matching between protocols and requirements
2. Intelligent LLM-based compliance analysis
3. Visual results display with charts and detailed breakdowns
4. Report generation and download functionality
"""

import streamlit as st
import tempfile
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

from src.processor import DocumentProcessor
from src.report_generator import ReportGenerator
from src.config import Config


# Page configuration
st.set_page_config(
    page_title="Clinical Trial Compliance Analyzer",
    page_icon="🏥",
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
        labels.append('✅ Followed')
        values.append(summary['followed'])
        colors.append('#28a745')
    
    if summary['partial'] > 0:
        labels.append('⚠️ Partial')
        values.append(summary['partial'])
        colors.append('#ffc107')
    
    if summary['not_followed'] > 0:
        labels.append('❌ Not Followed')
        values.append(summary['not_followed'])
        colors.append('#dc3545')
    
    if summary['not_applicable'] > 0:
        labels.append('➖ Not Applicable')
        values.append(summary['not_applicable'])
        colors.append('#6c757d')
    
    if summary.get('errors', 0) > 0:
        labels.append('⚠️ Errors')
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
    st.markdown("### 📊 Similarity Analysis Results")
    
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
        st.markdown("### 🔍 Top Matching Sections")
        
        for i, match in enumerate(results['top_matches'], 1):
            score = (1.0 - match.get('distance', 1.0)) * 100
            content = match.get('content', '')
            
            with st.expander(f"Match #{i} - Similarity: {score:.1f}%", expanded=(i == 1)):
                st.write(content)


def display_folder_results(best_match: dict, all_results: list):
    """Display results for folder matching mode"""
    st.markdown("### 🏆 Best Matching Protocol")
    
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
    st.markdown("### 📋 All Protocols Ranking")
    
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
    st.markdown("### 📊 Compliance Analysis Report")
    
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
    st.markdown("### 📈 Summary Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Requirements", summary['total_requirements'])
    
    with col2:
        st.metric("✅ Followed", summary['followed'])
    
    with col3:
        st.metric("⚠️ Partial", summary['partial'])
    
    with col4:
        st.metric("❌ Not Followed", summary['not_followed'])
    
    with col5:
        st.metric("➖ N/A", summary['not_applicable'])
    
    st.markdown("---")
    
    # Key findings
    st.markdown("### 🔍 Key Findings")
    
    if summary['not_followed'] > 0:
        st.error(f"⚠️ **{summary['not_followed']} requirement(s) not followed** - These require immediate attention.")
    
    if summary['partial'] > 0:
        st.warning(f"⚠️ **{summary['partial']} requirement(s) partially met** - These may need clarification or modification.")
    
    if summary['followed'] == summary['total_requirements']:
        st.success("✅ **Full compliance achieved** - All requirements are met by the protocol.")
    
    st.markdown("---")
    
    # Detailed requirements analysis
    st.markdown("### 📋 Detailed Requirements Analysis")
    
    # Group requirements by status
    status_groups = {
        "not_followed": {"title": "❌ Not Followed", "color": "red", "reqs": []},
        "partial": {"title": "⚠️ Partial", "color": "orange", "reqs": []},
        "followed": {"title": "✅ Followed", "color": "green", "reqs": []},
        "not_applicable": {"title": "➖ Not Applicable", "color": "gray", "reqs": []},
        "error": {"title": "⚠️ Error", "color": "red", "reqs": []}
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
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        options=["similarity", "intelligent"],
        format_func=lambda x: "🔍 Similarity Matching" if x == "similarity" else "🤖 Intelligent Analysis (LLM)",
        help="Similarity: Fast vector-based matching\nIntelligent: Detailed LLM-powered compliance analysis"
    )
    
    st.session_state.analysis_mode = analysis_mode
    
    # Show API key status for intelligent mode
    if analysis_mode == "intelligent":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔑 API Configuration")
        
        if Config.validate():
            st.sidebar.success(f"✅ {Config.LLM_PROVIDER.upper()} API configured")
            st.sidebar.caption(f"Model: {Config.get_llm_config()['model']}")
        else:
            st.sidebar.error(f"❌ {Config.LLM_PROVIDER.upper()}_API_KEY not found")
            st.sidebar.info("Please add your API key to the .env file")
    
    st.sidebar.markdown("---")
    
    # File upload section
    st.sidebar.markdown("## 📁 Upload Documents")
    
    # Protocol selection mode
    protocol_mode = st.sidebar.radio(
        "Protocol Mode",
        options=["single", "folder"],
        format_func=lambda x: "📄 Single Protocol" if x == "single" else "📂 Multiple Protocols (Folder)",
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
        "🚀 Start Analysis",
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


def main():
    """Main application logic"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🏥 Clinical Trial Compliance Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Match requirements to clinical trial protocols using vector similarity or intelligent LLM-based analysis</div>', unsafe_allow_html=True)
    
    # Sidebar
    sidebar_config = render_sidebar()
    
    # Main content area
    if not sidebar_config['analyze_button']:
        # Show welcome screen
        st.markdown("---")
        st.markdown("## 👋 Welcome!")
        st.markdown("""
        This application helps you analyze clinical trial protocol compliance with requirements documents.
        
        ### 🔍 Features:
        
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
        
        ### 📝 How to Use:
        
        1. **Configure** your analysis mode in the sidebar
        2. **Upload** your protocol and requirements documents
        3. **Click** "Start Analysis" to begin
        4. **Review** the results and download reports
        
        ### 📁 Supported File Types:
        - PDF (.pdf)
        - Text (.txt)
        - Markdown (.md)
        
        ---
        
        **Get Started:** Upload your documents in the sidebar! 👈
        """)
        
        # Show example scenarios
        st.markdown("### 💡 Example Use Cases:")
        
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
            st.error(f"❌ {Config.LLM_PROVIDER.upper()}_API_KEY not configured. Please add it to your .env file.")
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
                    
                    st.success("✅ Analysis complete!")
                    
                    display_compliance_results(results)
                    
                    # Generate and offer report download
                    st.markdown("---")
                    st.markdown("### 📄 Download Report")
                    
                    report_markdown = ReportGenerator.generate_markdown_report(results)
                    
                    st.download_button(
                        label="📥 Download Markdown Report",
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
                    
                    st.success(f"✅ Analysis complete! Best match: {results['protocol_name']}")
                    
                    # Show all protocols ranking first
                    if results.get('all_protocols'):
                        st.markdown("### 📋 Protocol Ranking")
                        
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
                    st.markdown("### 📄 Download Report")
                    
                    report_markdown = ReportGenerator.generate_markdown_report(results)
                    
                    st.download_button(
                        label="📥 Download Markdown Report",
                        data=report_markdown,
                        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
