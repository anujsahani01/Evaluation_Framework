"""
GitHub Extractor
================
Clones configured repos and extracts source files matching the configured extensions.
Produces a structured output of file metadata + content for the transform stage.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

from git import Repo

from src.config import get_config, PipelineConfig


@dataclass
class ExtractedFile:
    """Represents a single extracted source file."""

    repo_owner: str
    repo_name: str
    file_path: str          # Relative path within the repo
    language: str           # Detected from extension
    content: str
    size_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
    ".php": "php",
}


class GitHubExtractor:
    """
    Clones GitHub repos and extracts source files.

    Usage:
        extractor = GitHubExtractor()
        results = extractor.extract_all()
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or get_config()
        self.clone_dir = Path(self.config.extraction.clone_dir)
        self.output_dir = Path(self.config.extraction.output_dir)
        self.max_file_size = self.config.extraction.max_file_size_kb * 1024
        self._status: dict[str, str] = {}

    @property
    def status(self) -> dict[str, str]:
        """Get extraction status for all repos."""
        return self._status.copy()

    def extract_all(self) -> list[ExtractedFile]:
        """Extract source files from all configured repositories."""
        all_files: list[ExtractedFile] = []

        for repo_cfg in self.config.github.repos:
            repo_key = f"{repo_cfg.owner}/{repo_cfg.name}"
            self._status[repo_key] = "in_progress"

            try:
                files = self._extract_repo(repo_cfg.owner, repo_cfg.name, repo_cfg.branch)
                all_files.extend(files)
                self._status[repo_key] = f"completed ({len(files)} files)"
            except Exception as e:
                self._status[repo_key] = f"failed: {str(e)}"

        # Save extraction results
        self._save_results(all_files)
        return all_files

    def _extract_repo(self, owner: str, name: str, branch: str) -> list[ExtractedFile]:
        """Clone a single repo and extract matching files."""
        repo_url = self._build_clone_url(owner, name)
        local_path = self.clone_dir / owner / name

        # Clean previous clone if exists
        if local_path.exists():
            shutil.rmtree(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Clone the repo
        Repo.clone_from(repo_url, str(local_path), branch=branch, depth=1)

        # Walk and extract
        extracted: list[ExtractedFile] = []
        include_ext = set(self.config.github.include_extensions)
        exclude_dirs = set(self.config.github.exclude_dirs)

        for root, dirs, files in os.walk(local_path):
            # Filter excluded directories in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file_name in files:
                file_path = Path(root) / file_name
                ext = file_path.suffix.lower()

                if ext not in include_ext:
                    continue

                if file_path.stat().st_size > self.max_file_size:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                relative_path = str(file_path.relative_to(local_path))
                language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")

                extracted.append(
                    ExtractedFile(
                        repo_owner=owner,
                        repo_name=name,
                        file_path=relative_path,
                        language=language,
                        content=content,
                        size_bytes=len(content.encode("utf-8")),
                    )
                )

        return extracted

    def _build_clone_url(self, owner: str, name: str) -> str:
        """Build the clone URL, using token if available for private repos."""
        token = os.getenv("GITHUB_TOKEN", "")
        if token:
            return f"https://{token}@github.com/{owner}/{name}.git"
        return f"https://github.com/{owner}/{name}.git"

    def _save_results(self, files: list[ExtractedFile]) -> None:
        """Save extracted files as JSON for downstream consumption."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "extracted_files.json"

        data = [f.to_dict() for f in files]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
