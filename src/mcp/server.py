"""
MCP Server
==========
FastMCP server that exposes the AI pipeline as tools.

Tools:
- retrieval: Search the code knowledge base
- query_rewriter: Rewrite ambiguous queries for better retrieval
- answer_scorer: Score an answer's quality
- clarify_query: Determine if a query needs clarification
- process_query: Full query processing pipeline (rewrite → retrieve → generate)
- trigger_extraction: Trigger GitHub repo extraction
- trigger_transform: Trigger chunking of extracted files
- pipeline_status: Get status of extraction/chunking operations
"""

from __future__ import annotations

from fastmcp import FastMCP

from src.config import get_config, PipelineConfig
from src.extract.github_extractor import GitHubExtractor
from src.transform.chunker import CodeChunker
from src.embed.embedder import Embedder
from src.rag.pipeline import RAGPipeline


def create_mcp_server(config: PipelineConfig | None = None) -> FastMCP:
    """Create and configure the MCP server with all tools."""

    cfg = config or get_config()
    mcp = FastMCP("AI Pipeline MCP Server")

    # Shared state
    _extractor: GitHubExtractor | None = None
    _chunker: CodeChunker | None = None

    def _get_rag() -> RAGPipeline:
        return RAGPipeline(config=cfg)

    def _get_embedder() -> Embedder:
        return Embedder(config=cfg)

    # =========================================================================
    # TOOL: retrieval
    # =========================================================================
    @mcp.tool()
    def retrieval(query: str, top_k: int = 5) -> dict:
        """
        Search the code knowledge base using semantic similarity.

        Args:
            query: Natural language query about the codebase.
            top_k: Number of results to return.

        Returns:
            Dict with matched documents, metadata, and similarity scores.
        """
        embedder = _get_embedder()
        results = embedder.query(query_text=query, top_k=top_k)
        return {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
        }

    # =========================================================================
    # TOOL: query_rewriter
    # =========================================================================
    @mcp.tool()
    def query_rewriter(original_query: str) -> dict:
        """
        Rewrite an ambiguous or poorly-formed query into a better search query.

        Args:
            original_query: The user's original query.

        Returns:
            Dict with the rewritten query and explanation.
        """
        from src.llm_provider import generate
        import json

        response = generate(
            prompt=f"Original query: {original_query}",
            system_prompt=(
                "You are a query optimization expert. Rewrite the given query to be "
                "more specific and effective for searching a code knowledge base. "
                "Return ONLY a JSON object with keys 'rewritten_query' and 'explanation'."
            ),
            config=cfg,
        )

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {"rewritten_query": original_query, "explanation": "Could not parse rewrite."}

        return result

    # =========================================================================
    # TOOL: answer_scorer
    # =========================================================================
    @mcp.tool()
    def answer_scorer(query: str, answer: str, context: list[str]) -> dict:
        """
        Score an answer's quality based on the query and retrieved context.

        Args:
            query: The original user query.
            answer: The generated answer to score.
            context: The retrieval context used to generate the answer.

        Returns:
            Dict with scores for relevancy, faithfulness, and completeness.
        """
        from src.llm_provider import generate_as_judge
        import json

        context_str = "\n---\n".join(context)

        response = generate_as_judge(
            prompt=f"Query: {query}\n\nContext:\n{context_str}\n\nAnswer: {answer}",
            system_prompt=(
                "You are an answer quality evaluator. Score the answer on three dimensions:\n"
                "1. relevancy (0-1): Is the answer relevant to the query?\n"
                "2. faithfulness (0-1): Is the answer faithful to the provided context?\n"
                "3. completeness (0-1): Does the answer fully address the query?\n\n"
                "Return ONLY a JSON object with keys: relevancy, faithfulness, completeness, reasoning"
            ),
            config=cfg,
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"relevancy": 0, "faithfulness": 0, "completeness": 0, "reasoning": "Parse error"}

    # =========================================================================
    # TOOL: clarify_query
    # =========================================================================
    @mcp.tool()
    def clarify_query(query: str) -> dict:
        """
        Determine if a query needs clarification before processing.

        Args:
            query: The user's query to analyze.

        Returns:
            Dict with 'needs_clarification' boolean and optional 'questions' list.
        """
        from src.llm_provider import generate
        import json

        response = generate(
            prompt=f"Query: {query}",
            system_prompt=(
                "You analyze queries about codebases and determine if they need clarification. "
                "A query needs clarification if it is ambiguous, too broad, or missing key details. "
                "Return ONLY JSON with keys: 'needs_clarification' (bool), 'questions' (list of "
                "clarifying questions if needed), 'confidence' (0-1 how confident you are the "
                "query is clear enough)."
            ),
            config=cfg,
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"needs_clarification": False, "questions": [], "confidence": 0.5}

    # =========================================================================
    # TOOL: process_query
    # =========================================================================
    @mcp.tool()
    def process_query(query: str) -> dict:
        """
        Full query processing pipeline: rewrite → retrieve → generate.

        Args:
            query: User's natural language question about the codebase.

        Returns:
            Dict with the answer, retrieved context, and intermediate results.
        """
        rag = _get_rag()
        rag_response = rag.query(query)

        return {
            "query": rag_response.query,
            "answer": rag_response.answer,
            "retrieval_context": rag_response.retrieval_context,
            "num_chunks_retrieved": len(rag_response.retrieval_context),
            "similarity_scores": rag_response.scores,
        }

    # =========================================================================
    # TOOL: trigger_extraction
    # =========================================================================
    @mcp.tool()
    def trigger_extraction() -> dict:
        """
        Trigger extraction of source code from configured GitHub repositories.

        Returns:
            Dict with extraction status and file counts.
        """
        nonlocal _extractor
        _extractor = GitHubExtractor(config=cfg)
        files = _extractor.extract_all()

        return {
            "status": "completed",
            "total_files_extracted": len(files),
            "repo_status": _extractor.status,
        }

    # =========================================================================
    # TOOL: trigger_transform
    # =========================================================================
    @mcp.tool()
    def trigger_transform() -> dict:
        """
        Trigger chunking of previously extracted files, then embed and store.

        Returns:
            Dict with chunking results and embedding status.
        """
        nonlocal _chunker
        import json
        from pathlib import Path

        # Load extracted files
        extracted_path = Path(cfg.extraction.output_dir) / "extracted_files.json"
        if not extracted_path.exists():
            return {"status": "error", "message": "No extracted files found. Run extraction first."}

        from src.extract.github_extractor import ExtractedFile

        with open(extracted_path, "r") as f:
            raw = json.load(f)

        extracted_files = [ExtractedFile(**item) for item in raw]

        # Chunk
        _chunker = CodeChunker(config=cfg)
        chunks = _chunker.chunk_all(extracted_files)

        # Embed and store
        embedder = Embedder(config=cfg)
        stored_count = embedder.embed_and_store(chunks)

        return {
            "status": "completed",
            "total_chunks": len(chunks),
            "chunks_stored": stored_count,
            "chunker_status": _chunker.status,
        }

    # =========================================================================
    # TOOL: pipeline_status
    # =========================================================================
    @mcp.tool()
    def pipeline_status() -> dict:
        """
        Get the current status of extraction and chunking operations.

        Returns:
            Dict with status info for each pipeline component.
        """
        embedder = _get_embedder()
        stats = embedder.get_collection_stats()

        return {
            "extraction": _extractor.status if _extractor else "not_started",
            "chunking": _chunker.status if _chunker else "not_started",
            "vector_db": stats,
        }

    return mcp


# --- Entry point for running the MCP server ---
if __name__ == "__main__":
    config = get_config()
    server = create_mcp_server(config)
    server.run()
