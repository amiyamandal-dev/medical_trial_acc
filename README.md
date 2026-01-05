# Clinical Trial Protocol Matcher

A powerful CLI tool that matches requirements documents to clinical trial protocol documents using **vector similarity** or **intelligent LLM-based analysis powered by LangChain**.

## Features

- **🚀 Two Modes**: Fast vector similarity OR intelligent LLM analysis
- **🧠 LangChain Integration**: Structured LLM chains for requirement extraction and analysis
- **📊 Detailed Compliance Reports**: Status, confidence scores, and explanations for each requirement
- **📁 Smart Matching**: Single file or automatic best-match from folder
- **📄 Multiple Formats**: Supports PDF, TXT, and Markdown files
- **🎨 Beautiful CLI**: Rich terminal UI with progress tracking and colored output

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Configuration

For intelligent analysis mode, you need a DeepSeek API key:

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com  # Optional
DEEPSEEK_MODEL=deepseek-reasoner             # Optional
```

## Quick Start

### Basic Mode (No API Key Required)

Fast vector similarity matching:

```bash
# Match single protocol
python main.py data/protocols/diabetes.pdf data/requirements/req.txt

# Find best protocol from folder
python main.py data/protocols/ data/requirements/req.txt
```

### Intelligent Mode (Requires API Key)

LLM-powered compliance analysis:

```bash
# Analyze single protocol
python main.py data/protocols/diabetes.pdf data/requirements/req.txt --analyze

# Find best protocol and analyze
python main.py data/protocols/ data/requirements/req.txt --analyze
```

## Usage Guide

### Mode 1: Vector Similarity (Basic)

Fast matching using semantic embeddings - perfect for quickly finding relevant protocols.

**Use when:**
- You want fast results
- No API key available
- Just need to find similar documents
- Budget-conscious

**What you get:**
- Similarity scores (0-100%)
- Best matching protocol identification
- Top matching sections
- Protocol rankings

**Example:**
```bash
python main.py data/protocols/ data/requirements/diabetes_req.txt
```

**Output:**
```
Mode: Find Best Protocol from Folder (Vector Similarity)

╭─────────── Best Match ───────────╮
│ ✓ Best Matching Protocol:        │
│                                   │
│ File: diabetes_phase3.txt         │
│ Similarity Score: 82.5%           │
│ Max Section Score: 89.3%          │
│ Matching Sections: 8              │
╰───────────────────────────────────╯

All Protocol Rankings:
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Rank ┃ Protocol Name     ┃ Avg Score ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ #1   │ diabetes_phase3.. │     82.5% │
│ #2   │ oncology_phase2.. │     34.2% │
│ #3   │ cardiology_p...   │     28.7% │
└──────┴───────────────────┴───────────┘
```

### Mode 2: Intelligent Analysis (LLM-Powered)

Deep compliance analysis using LangChain and LLM reasoning - perfect for regulatory compliance.

**Use when:**
- Need detailed compliance analysis
- Want explanations for each requirement
- Checking regulatory compliance
- Quality assurance reviews

**What you get:**
- Individual requirements extracted automatically
- Compliance status for each: ✅ Followed, ⚠️ Partial, ❌ Not Followed, ➖ N/A
- Confidence scores (0-100%)
- Detailed explanations from LLM
- Overall compliance rate
- **Markdown report automatically saved to `reports/` folder**

**Example:**
```bash
python main.py data/protocols/diabetes.pdf data/requirements/req.txt --analyze
```

**Output:**
```
Mode: Intelligent Compliance Analysis (LLM-based)

  Extracting requirements... ━━━━━━━━━━━━━━━━━━━━━━━━ 10%
  Found 12 requirements ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10%
  Analyzing requirement 1/12 ━━━━━━━━━━━━━━━━━━━━━ 17%
  ...
  Analysis complete! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

📊 Compliance Analysis Report
Protocol: diabetes_phase3.pdf
Requirements: diabetes_requirements.txt

Summary
────────────────────────────────────
Total Requirements      12
✅ Followed             8
⚠️  Partial             3
❌ Not Followed         1
➖ Not Applicable       0

Compliance Rate         91.7%

Detailed Requirements Analysis
────────────────────────────────────

1. ✅ Followed
   Requirement: Primary endpoint must be HbA1c reduction from baseline
   Confidence: 95%
   Explanation: The protocol explicitly defines HbA1c reduction as the
   primary endpoint, measured at 12 and 24 weeks with a target change
   of at least 0.5% from baseline.

2. ⚠️  Partial
   Requirement: Safety monitoring must include adverse event reporting
   within 24 hours for serious events
   Confidence: 80%
   Explanation: Protocol includes comprehensive AE reporting but specifies
   48-hour timeline for serious events, which partially meets the requirement.

3. ❌ Not Followed
   Requirement: Study must include pediatric population aged 12-17
   Confidence: 90%
   Explanation: Protocol explicitly excludes participants under 18 years
   of age, directly contradicting this requirement.

📄 Report saved to: reports/diabetes_requirements_analysis_20260105_120130.md
```

### Markdown Report Output

Analysis results are automatically saved to a comprehensive markdown report in the `reports/` folder:

- **Executive Summary** with compliance rate and badges
- **Summary Table** with percentages
- **Key Findings** highlighting critical issues
- **Detailed Analysis** grouped by compliance status (priority order)
- **Recommendations** for addressing gaps
- **Metadata** about the analysis

The report can be:
- Viewed in any markdown viewer
- Converted to PDF
- Shared with stakeholders
- Version controlled in git
```

## How It Works

### Basic Mode Architecture

```
Documents → PDF/Text Extraction → Text Chunking (LangChain)
                                         ↓
                                  Vector Embeddings
                                         ↓
                                  ChromaDB Search
                                         ↓
                               Similarity Scoring
```

**Technologies:**
- **LangChain TextSplitters**: Intelligent document chunking
- **sentence-transformers**: Embedding generation (`all-MiniLM-L6-v2`)
- **ChromaDB**: Vector storage and similarity search
- **Rich**: Beautiful terminal output

### Intelligent Mode Architecture

```
Requirements Document → Regex Parsing → Individual Requirements
         ↓                                      ↓
    (If needed)                          For each requirement:
         ↓                                      ↓
LangChain LLM Extraction              Vector Search (ChromaDB)
         ↓                                      ↓
   Requirements List                   Top Protocol Sections
                                               ↓
                                    LangChain Analysis Chain
                                               ↓
                              LLM Reasoning (DeepSeek-R1)
                                               ↓
                            Status + Confidence + Explanation
```

**Technologies:**
- **LangChain**: Structured LLM chains and prompts
- **ChatOpenAI**: LLM interface (DeepSeek compatible)
- **ChatPromptTemplate**: Consistent prompt engineering
- **StrOutputParser**: Response parsing
- **Vector Search**: Context retrieval for each requirement

### LangChain Integration Benefits

1. **Structured Prompts**: Consistent, maintainable prompt templates
2. **Chains**: Composable LLM operations with `|` operator
3. **Type Safety**: Better IDE support and error handling
4. **Modularity**: Easy to swap models or modify prompts
5. **Best Practices**: Industry-standard LLM orchestration

## Project Structure

```
medical-trial-acc/
├── main.py                  # CLI entry point with arg parsing
├── src/
│   ├── agent.py             # LangChain agent (chains, prompts, Pydantic)
│   ├── processor.py         # Document processing orchestration
│   ├── vector_store.py      # ChromaDB + LangChain splitters
│   ├── report_generator.py  # Markdown report generation
│   └── config.py            # Environment configuration
├── data/
│   ├── protocols/           # Place protocol PDFs/TXT here
│   └── requirements/        # Place requirements docs here
├── reports/                 # Generated markdown reports (auto-created)
├── .env                     # API keys (from .env.example)
├── pyproject.toml           # uv dependencies
└── README.md                # This file
```

## Technical Details

### Vector Store
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Database**: ChromaDB with cosine similarity
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Chunk Size**: 1000 characters with 200 overlap
- **Storage**: Local ChromaDB in `data/chroma_db/`

### LLM Agent (LangChain)
- **Framework**: LangChain core with chains
- **Provider**: DeepSeek API (OpenAI-compatible)
- **Model**: `deepseek-reasoner` (configurable)
- **Temperature**: 0.1 (focused, deterministic)
- **Max Tokens**: 500 per requirement
- **Chains**:
  - Extract Requirements Chain
  - Analyze Requirement Chain

### Requirements Extraction
1. **Primary**: Fast regex parsing (`^\d+\.\d+\s+` pattern)
2. **Fallback**: LangChain LLM extraction chain
3. **Filters**: Removes headers, metadata, short items

## Supported File Types

| Extension | Type | Notes |
|-----------|------|-------|
| `.pdf` | PDF documents | Text extraction via pypdf |
| `.txt` | Plain text | Direct reading |
| `.md` | Markdown | Direct reading |

## Similarity/Compliance Scoring

### Basic Mode (Similarity Scores)
- 🟢 **70-100%**: High similarity - excellent match
- 🟡 **40-69%**: Medium similarity - good match
- 🔴 **0-39%**: Low similarity - weak match

### Intelligent Mode (Compliance Status)
- ✅ **Followed**: Requirement fully met by protocol
- ⚠️  **Partial**: Requirement partially addressed
- ❌ **Not Followed**: Requirement not met or contradicted
- ➖ **Not Applicable**: Requirement doesn't apply

## Performance Comparison

| Metric | Basic Mode | Intelligent Mode |
|--------|------------|------------------|
| **Speed** | ~2-5 sec | ~10-60 sec |
| **Cost** | Free | ~$0.01-0.10/analysis |
| **Accuracy** | Good for similarity | Excellent for compliance |
| **Detail Level** | Scores only | Full explanations |
| **Use Case** | Quick screening | Regulatory review |

## Command Reference

```bash
# Help
python main.py --help
python main.py -h

# Basic similarity matching
python main.py <protocol_file> <requirements_file>
python main.py <protocols_folder> <requirements_file>

# Intelligent analysis
python main.py <protocol_file> <requirements_file> --analyze
python main.py <protocols_folder> <requirements_file> --analyze

# With debug output
python main.py <protocol_file> <requirements_file> --analyze --debug
```

## Examples

### Example 1: Quick Protocol Search
```bash
python main.py data/protocols/ data/requirements/diabetes_req.txt
```
**Result**: Instantly find the best matching protocol from a folder

### Example 2: Detailed Compliance Check
```bash
python main.py data/protocols/diabetes_phase3.pdf \
               data/requirements/diabetes_req.txt \
               --analyze
```
**Result**: Full compliance report with requirement-by-requirement analysis

### Example 3: Batch Processing (Script)
```bash
#!/bin/bash
# Batch analyze all requirements documents
for req in data/requirements/*.txt; do
  echo "Analyzing: $req"
  python main.py data/protocols/ "$req" --analyze
  # Reports are automatically saved to reports/ folder
done

# All markdown reports will be in reports/ directory
ls -l reports/
```

## Tips & Best Practices

### For Better Results

1. **Document Quality**
   - Use clear, well-structured documents
   - Numbered requirements work best
   - Avoid scanned PDFs (OCR needed)

2. **Requirements Writing**
   - One requirement per statement
   - Use specific, measurable criteria
   - Avoid ambiguous language

3. **Mode Selection**
   - Use **Basic Mode** for initial screening
   - Use **Intelligent Mode** for final compliance checks
   - Combine both: Basic → find candidates → Intelligent → detailed analysis

4. **Performance**
   - Basic mode handles 100s of documents easily
   - Intelligent mode: limit to most relevant protocols first
   - Set `TOKENIZERS_PARALLELISM=false` to suppress warnings

### LangChain Customization

To modify prompts or chains, edit `src/agent.py`:

```python
# Example: Customize analysis prompt
self.analyze_requirement_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert clinical trial analyst..."),
    ("user", "Your custom prompt template here...")
])
```

## Troubleshooting

### Issue: "DEEPSEEK_API_KEY not found"
**Solution**: Create `.env` file or run without `--analyze` flag

### Issue: Slow analysis
**Solution**: Normal for intelligent mode (LLM calls). Each requirement takes ~1-3 seconds.

### Issue: Poor extraction of requirements
**Solution**:
- Ensure requirements are numbered (1.1, 1.2, etc.)
- Or let LLM extraction handle it (automatic fallback)

### Issue: ChromaDB errors
**Solution**: Delete `data/chroma_db/` and run again

### Issue: Import errors
**Solution**: Run `uv sync` to install all dependencies

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key | - | For `--analyze` |
| `DEEPSEEK_BASE_URL` | API base URL | `https://api.deepseek.com` | No |
| `DEEPSEEK_MODEL` | Model name | `deepseek-reasoner` | No |
| `TOKENIZERS_PARALLELISM` | Disable tokenizer warnings | - | No (set to `false`) |

## Output Formats

### Terminal Display
- Rich formatted output with colors and tables
- Progress bars for long-running analyses
- Expandable sections

### Markdown Reports
- Comprehensive reports saved to `reports/` folder
- Professional formatting with tables and badges
- Grouped by compliance status (priority order)
- Includes recommendations and metadata
- Can be converted to PDF or shared directly

**Report Features:**
- Executive summary with compliance badges
- Detailed breakdown by requirement
- Key findings and recommendations
- Full audit trail with timestamps
- Ready for stakeholder review

## Dependencies

Key dependencies managed by `uv`:
- `langchain` - LLM orchestration framework
- `langchain-openai` - OpenAI/DeepSeek integration
- `langchain-text-splitters` - Document chunking
- `pydantic` - Response validation and type safety
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `pypdf` - PDF text extraction
- `rich` - Beautiful terminal UI
- `python-dotenv` - Environment management

## Contributing

This project uses:
- **uv** for dependency management
- **LangChain** for LLM operations
- **Rich** for CLI interface
- **ChromaDB** for vector storage

## License

See LICENSE file for details.

## Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/) - LLM application framework
- [DeepSeek](https://www.deepseek.com/) - Reasoning LLM
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
- [sentence-transformers](https://www.sbert.net/) - Embedding models
