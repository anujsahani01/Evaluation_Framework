"""
Configuration loader for the AI Pipeline.
Reads pipeline_config.yaml and provides typed access to all settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file if present
load_dotenv()

# Project root is the parent of this file's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"


# --- Pydantic Models for typed config ---


class LLMConfig(BaseModel):
    provider: str = "huggingface"
    model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    judge_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    api_key_env: str = "HUGGINGFACE_API_KEY"
    base_url: str | None = None         # Override API base URL (null = use provider default)
    temperature: float = 0.0
    max_tokens: int = 4096

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise ValueError(
                f"API key not found. Set the '{self.api_key_env}' environment variable.\n"
                f"Provider: {self.provider}, Model: {self.model}"
            )
        return key


class GitHubRepoConfig(BaseModel):
    owner: str
    name: str
    branch: str = "main"


class GitHubConfig(BaseModel):
    repos: list[GitHubRepoConfig] = Field(default_factory=list)
    include_extensions: list[str] = Field(default_factory=lambda: [".py"])
    exclude_dirs: list[str] = Field(default_factory=lambda: ["node_modules", ".git", "__pycache__"])


class ExtractionConfig(BaseModel):
    clone_dir: str = "./data/repos"
    output_dir: str = "./data/extracted"
    max_file_size_kb: int = 500


class ChunkingConfig(BaseModel):
    strategy: str = "code_aware"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    language_parsers: dict[str, bool] = Field(default_factory=lambda: {"python": True})


class EmbeddingConfig(BaseModel):
    provider: str = "huggingface"       # openai | huggingface | ollama (INDEPENDENT of llm.provider)
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = 32


class VectorDBConfig(BaseModel):
    provider: str = "chromadb"
    persist_dir: str = "./data/chromadb"
    collection_name: str = "source_code"
    distance_metric: str = "cosine"


class RAGConfig(BaseModel):
    top_k: int = 5
    rerank: bool = True
    rerank_model: str | None = None     # None = use same model as llm.model
    similarity_threshold: float = 0.7


class MCPConfig(BaseModel):
    host: str = "localhost"
    port: int = 8000
    tools: list[str] = Field(default_factory=list)


class EvalThresholds(BaseModel):
    context_recall: float = 0.7
    context_precision: float = 0.7
    context_relevancy: float = 0.7
    faithfulness: float = 0.8
    answer_relevancy: float = 0.7
    tool_correctness: float = 0.8
    task_completion: float = 0.7
    step_efficiency: float = 0.6
    argument_correctness: float = 0.7
    plan_adherence: float = 0.7


class EvaluationConfig(BaseModel):
    thresholds: EvalThresholds = Field(default_factory=EvalThresholds)
    golden_dataset_path: str = "./eval/datasets/golden_dataset.json"
    synthetic_test_count: int = 20


class PipelineConfig(BaseModel):
    """Root configuration model for the entire pipeline."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    """Load and validate the pipeline configuration."""
    path = Path(config_path) if config_path else CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return PipelineConfig(**raw)


# Singleton config instance (lazy loaded)
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    """Get the singleton config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
