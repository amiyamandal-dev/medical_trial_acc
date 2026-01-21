#!/bin/bash
# Quick start script for Clinical Trial Compliance Analyzer Streamlit App

echo "🏥 Starting Clinical Trial Compliance Analyzer..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   For intelligent analysis, copy .env.example to .env and add your API keys"
    echo ""
fi

# Run Streamlit
echo "🚀 Launching Streamlit interface on http://localhost:8501"
echo ""

streamlit run app.py
