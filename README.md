# AI Pipeline Evaluation

An AI Systems Engineering project that builds a complete code-understanding pipeline with **DeepEval-powered CI/CD evaluation** at every component level.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   EXTRACT   │────▶│  TRANSFORM  │────▶│    EMBED    │
│  (GitHub)   │     │  (Chunking) │     │  (ChromaDB) │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    EVAL     │◀────│     MCP     │◀────│     RAG     │
│  (DeepEval) │     │  (FastMCP)  │     │  (Pipeline) │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Components

| Component | Purpose | Eval Metrics |
|-----------|---------|--------------|
| Extract | Pull source from GitHub repos | Completeness, coverage |
| Transform | Code-aware chunking | Coherence, boundary preservation, info density |
| Embed | Vector embedding + ChromaDB storage | - |
| RAG | Retrieval + Generation pipeline | Context precision/recall/relevancy, faithfulness, answer relevancy |
| MCP | Tool server (FastMCP) | Tool correctness, task completion, step efficiency |
| Eval | DeepEval unit tests | Meta: validates all above |

## Quick Start

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .

# 2. Configure
cp .env.example .env          # Add your OPENAI_API_KEY
# Edit config/pipeline_config.yaml with your repos

# 3. Run the pipeline
python -m src.mcp.server      # Start MCP server

# 4. Run evaluations (CI/CD gate)
deepeval test run eval/
```

## Evaluation Philosophy

This isn't "run ROUGE and call it a day." Every component is evaluated:

- **Chunking**: Are chunks semantically coherent? Do they respect code boundaries?
- **Retrieval**: Is the right context fetched? Is it ranked correctly?
- **Generation**: Is the answer faithful? Relevant? Actually helpful for a developer?
- **Agent**: Did it pick the right tool? Was it efficient? Did it complete the task?

All metrics have configurable thresholds in `config/pipeline_config.yaml`. Tests **fail** if scores drop below thresholds — just like unit tests gate deployment.

## Running Evaluations

```bash
# Run all eval tests
deepeval test run eval/

# Run specific component
deepeval test run eval/test_retrieval.py
deepeval test run eval/test_generation.py
deepeval test run eval/test_chunking.py
deepeval test run eval/test_agent.py
deepeval test run eval/test_e2e_pipeline.py

# With verbose output
deepeval test run eval/ -v
```

## Tech Stack

- **Python 3.11+**
- **DeepEval 4.0** - Evaluation framework with 50+ metrics
- **ChromaDB** - Vector database
- **FastMCP** - MCP server framework
- **OpenAI** - LLM provider (configurable)
- **LangChain Text Splitters** - Code-aware chunking
