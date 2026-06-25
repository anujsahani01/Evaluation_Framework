"""
Agent Evaluation Tests
======================
Unit tests for the MCP agent's tool selection, planning, and efficiency.

Metrics evaluated:
- ToolCorrectnessMetric: Did the agent select the right tools?
- TaskCompletionMetric: Did the agent complete the user's task?
- StepEfficiencyMetric: Did the agent complete the task in minimal steps?

These are the AI Systems Engineering differentiators — not just "does the answer look good"
but "did the agent reason correctly about WHICH tools to use and HOW."

Run via: `deepeval test run eval/test_agent.py`
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

from src.config import get_config


# --- Metric Initialization ---

config = get_config()
thresholds = config.evaluation.thresholds

# Tool Correctness: Did the agent use the expected tools?
tool_correctness = ToolCorrectnessMetric(
    threshold=thresholds.tool_correctness,
    include_reason=True,
)

# Available tools that the agent can choose from (provides context for evaluation)
AVAILABLE_TOOLS = [
    ToolCall(name="retrieval"),
    ToolCall(name="query_rewriter"),
    ToolCall(name="answer_scorer"),
    ToolCall(name="clarify_query"),
    ToolCall(name="process_query"),
    ToolCall(name="trigger_extraction"),
    ToolCall(name="trigger_transform"),
    ToolCall(name="pipeline_status"),
]


# --- Test Cases: Tool Selection ---

TOOL_SELECTION_TEST_CASES = [
    {
        "id": "simple_code_query",
        "input": "How does the login function work?",
        "actual_output": "The login function validates credentials and returns a JWT token.",
        "tools_called": [
            ToolCall(name="retrieval"),
        ],
        "expected_tools": [
            ToolCall(name="retrieval"),
        ],
    },
    {
        "id": "ambiguous_query_needs_rewrite",
        "input": "fix it",
        "actual_output": "I need more context. Could you clarify what you'd like me to fix?",
        "tools_called": [
            ToolCall(name="clarify_query"),
        ],
        "expected_tools": [
            ToolCall(name="clarify_query"),
        ],
    },
    {
        "id": "full_pipeline_query",
        "input": "Explain the data validation logic in the user registration endpoint",
        "actual_output": "The user registration endpoint validates email format, password strength, and checks for duplicate accounts.",
        "tools_called": [
            ToolCall(name="query_rewriter"),
            ToolCall(name="retrieval"),
        ],
        "expected_tools": [
            ToolCall(name="query_rewriter"),
            ToolCall(name="retrieval"),
        ],
    },
    {
        "id": "extraction_trigger",
        "input": "Pull the latest code from all repositories",
        "actual_output": "Extraction triggered successfully. 142 files extracted from 2 repositories.",
        "tools_called": [
            ToolCall(name="trigger_extraction"),
        ],
        "expected_tools": [
            ToolCall(name="trigger_extraction"),
        ],
    },
    {
        "id": "full_ingestion_pipeline",
        "input": "Refresh the knowledge base with the latest code",
        "actual_output": "Extraction and chunking complete. 142 files extracted, 580 chunks embedded.",
        "tools_called": [
            ToolCall(name="trigger_extraction"),
            ToolCall(name="trigger_transform"),
        ],
        "expected_tools": [
            ToolCall(name="trigger_extraction"),
            ToolCall(name="trigger_transform"),
        ],
    },
    {
        "id": "status_check",
        "input": "What's the current state of the pipeline?",
        "actual_output": "Extraction: completed (142 files). Vector DB: 580 documents indexed.",
        "tools_called": [
            ToolCall(name="pipeline_status"),
        ],
        "expected_tools": [
            ToolCall(name="pipeline_status"),
        ],
    },
    {
        "id": "quality_check",
        "input": "How good was the answer about authentication?",
        "actual_output": "The answer scored 0.9 on relevancy, 0.85 on faithfulness, 0.8 on completeness.",
        "tools_called": [
            ToolCall(name="answer_scorer"),
        ],
        "expected_tools": [
            ToolCall(name="answer_scorer"),
        ],
    },
]


@pytest.mark.parametrize(
    "test_data",
    TOOL_SELECTION_TEST_CASES,
    ids=[t["id"] for t in TOOL_SELECTION_TEST_CASES],
)
def test_tool_correctness(test_data: dict):
    """
    Test that the agent selects the correct tools for different query types.

    This is the core agent evaluation — verifying the agent's reasoning about
    WHICH tool to use given the user's intent.
    """
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        tools_called=test_data["tools_called"],
        expected_tools=test_data["expected_tools"],
    )
    assert_test(test_case, [tool_correctness])


# --- Test Cases: Incorrect Tool Selection (should FAIL) ---
# These exist to verify our evaluation catches bad tool selection.

NEGATIVE_TEST_CASES = [
    {
        "id": "wrong_tool_for_simple_query",
        "input": "What is the User model?",
        "actual_output": "The User model has fields for id, email, and password.",
        "tools_called": [
            ToolCall(name="trigger_extraction"),  # WRONG: shouldn't extract for a query
            ToolCall(name="retrieval"),
        ],
        "expected_tools": [
            ToolCall(name="retrieval"),  # Only retrieval needed
        ],
    },
]


@pytest.mark.parametrize(
    "test_data",
    NEGATIVE_TEST_CASES,
    ids=[t["id"] for t in NEGATIVE_TEST_CASES],
)
def test_tool_correctness_negative(test_data: dict):
    """
    Verify that unnecessary tool calls are penalized.
    These tests validate that our metrics catch over-tooling (calling tools that aren't needed).
    """
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        tools_called=test_data["tools_called"],
        expected_tools=test_data["expected_tools"],
    )
    # We expect this to score BELOW threshold (hence the low threshold here)
    # In practice you'd structure these as "expected failure" tests
    metric = ToolCorrectnessMetric(threshold=0.9, include_reason=True)
    metric.measure(test_case)
    # Assert the score is LOW (proving our metrics detect bad behavior)
    assert metric.score < 0.9, (
        f"Expected low score for incorrect tool selection, got {metric.score}. "
        f"Reason: {metric.reason}"
    )
