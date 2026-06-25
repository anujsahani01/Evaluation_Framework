"""
Code-Aware Chunker
==================
Splits extracted source code into semantically meaningful chunks.
Uses language-aware splitting that respects function/class boundaries
rather than naively splitting on character count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from src.config import get_config, PipelineConfig
from src.extract.github_extractor import ExtractedFile


LANGUAGE_MAP: dict[str, Language] = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "java": Language.JAVA,
    "go": Language.GO,
    "rust": Language.RUST,
    "cpp": Language.CPP,
    "c": Language.C,
    "ruby": Language.RUBY,
    "php": Language.PHP,
}


@dataclass
class CodeChunk:
    """A single chunk of code with metadata for embedding."""

    chunk_id: str
    repo_owner: str
    repo_name: str
    file_path: str
    language: str
    content: str
    chunk_index: int            # Position in the file's chunk sequence
    total_chunks: int           # Total chunks for this file
    start_line: int | None      # Approximate start line (if available)
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def metadata(self) -> dict:
        """Metadata to store alongside the embedding in vector DB."""
        return {
            "chunk_id": self.chunk_id,
            "repo": f"{self.repo_owner}/{self.repo_name}",
            "file_path": self.file_path,
            "language": self.language,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


class CodeChunker:
    """
    Transforms extracted files into chunks suitable for embedding.

    Supports strategies:
    - code_aware: Uses language-specific separators (function/class boundaries)
    - recursive: Generic recursive character splitting
    - fixed_size: Fixed token count chunks
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or get_config()
        self.chunk_size = self.config.chunking.chunk_size
        self.chunk_overlap = self.config.chunking.chunk_overlap
        self.strategy = self.config.chunking.strategy
        self._status: dict[str, str] = {}

    @property
    def status(self) -> dict[str, str]:
        return self._status.copy()

    def chunk_all(self, extracted_files: list[ExtractedFile]) -> list[CodeChunk]:
        """Chunk all extracted files."""
        all_chunks: list[CodeChunk] = []

        for file in extracted_files:
            file_key = f"{file.repo_owner}/{file.repo_name}/{file.file_path}"
            self._status[file_key] = "chunking"

            try:
                chunks = self._chunk_file(file)
                all_chunks.extend(chunks)
                self._status[file_key] = f"done ({len(chunks)} chunks)"
            except Exception as e:
                self._status[file_key] = f"failed: {str(e)}"

        # Save chunks
        self._save_chunks(all_chunks)
        return all_chunks

    def _chunk_file(self, file: ExtractedFile) -> list[CodeChunk]:
        """Chunk a single file using the configured strategy."""
        splitter = self._get_splitter(file.language)
        texts = splitter.split_text(file.content)

        chunks: list[CodeChunk] = []
        for idx, text in enumerate(texts):
            chunk_id = f"{file.repo_owner}/{file.repo_name}/{file.file_path}::chunk_{idx}"
            chunk = CodeChunk(
                chunk_id=chunk_id,
                repo_owner=file.repo_owner,
                repo_name=file.repo_name,
                file_path=file.file_path,
                language=file.language,
                content=text,
                chunk_index=idx,
                total_chunks=len(texts),
                start_line=self._estimate_start_line(file.content, text),
                token_count=len(text.split()),  # Rough estimate; use tiktoken for precision
            )
            chunks.append(chunk)

        return chunks

    def _get_splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        """Get the appropriate text splitter based on strategy and language."""
        if self.strategy == "code_aware" and language in LANGUAGE_MAP:
            return RecursiveCharacterTextSplitter.from_language(
                language=LANGUAGE_MAP[language],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        # Fallback to recursive splitting
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _estimate_start_line(self, full_content: str, chunk_content: str) -> int | None:
        """Estimate the starting line number of a chunk within the full file."""
        idx = full_content.find(chunk_content[:100])  # Match on first 100 chars
        if idx == -1:
            return None
        return full_content[:idx].count("\n") + 1

    def _save_chunks(self, chunks: list[CodeChunk]) -> None:
        """Save chunks to disk for inspection/debugging."""
        output_dir = Path(self.config.extraction.output_dir).parent / "chunks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "chunks.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
