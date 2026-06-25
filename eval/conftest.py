"""
Shared pytest fixtures for the evaluation suite.
Provides configured pipeline components and test data for all eval tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Ensure env vars are loaded
load_dotenv()

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, PipelineConfig


@pytest.fixture(scope="session")
def pipeline_config() -> PipelineConfig:
    """Session-scoped pipeline configuration."""
    return get_config()


@pytest.fixture(scope="session")
def golden_dataset_path(pipeline_config: PipelineConfig) -> Path:
    """Path to the golden evaluation dataset."""
    return Path(pipeline_config.evaluation.golden_dataset_path)


@pytest.fixture(scope="session")
def golden_dataset(golden_dataset_path: Path) -> list[dict]:
    """Load the golden evaluation dataset."""
    if not golden_dataset_path.exists():
        pytest.skip("Golden dataset not found. Generate it first with `python -m eval.generate_dataset`")
    with open(golden_dataset_path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def eval_thresholds(pipeline_config: PipelineConfig) -> dict[str, float]:
    """Evaluation threshold values from config."""
    return pipeline_config.evaluation.thresholds.model_dump()
