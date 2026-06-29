"""
LLM Provider Abstraction
=========================
Unified interface for switching between LLM providers (OpenAI, HuggingFace, Ollama, etc.)
by just changing the config. No code changes needed.

Supports:
- openai: GPT-4o-mini, GPT-4o, etc.
- huggingface: Any model on HF Inference API (Mistral, Llama, Phi, etc.)
- ollama: Local models via Ollama
- any provider that follows OpenAI-compatible API (Groq, Together, etc.)

Usage:
    from src.llm_provider import get_llm_client, generate, embed_texts

    # Generate text
    response = generate("What is this function?", system_prompt="You are a code expert.")

    # Get embeddings
    vectors = embed_texts(["def hello(): pass"])
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.config import get_config, PipelineConfig


class LLMProviderType(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    COMPATIBLE = "compatible"  # Any OpenAI-compatible API (Groq, Together, etc.)


class LLMResponse(BaseModel):
    """Unified response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: dict[str, int] = Field(default_factory=dict)  # tokens used


class EmbeddingResponse(BaseModel):
    """Unified embedding response."""
    embeddings: list[list[float]]
    model: str
    provider: str


# =============================================================================
# Provider Implementations
# =============================================================================


def _get_api_key(config: PipelineConfig) -> str:
    """Get the API key from environment variable specified in config."""
    key_env = config.llm.api_key_env
    key = os.getenv(key_env, "")
    if not key:
        raise ValueError(
            f"API key not found. Set the '{key_env}' environment variable.\n"
            f"Provider: {config.llm.provider}, Model: {config.llm.model}"
        )
    return key


def _generate_openai(
    prompt: str,
    system_prompt: str | None,
    config: PipelineConfig,
    model_override: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Generate using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=_get_api_key(config))
    model = model_override or config.llm.model

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=kwargs.get("temperature", config.llm.temperature),
        max_tokens=kwargs.get("max_tokens", config.llm.max_tokens),
    )

    return LLMResponse(
        content=response.choices[0].message.content or "",
        model=model,
        provider="openai",
        usage={
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        },
    )


def _generate_huggingface(
    prompt: str,
    system_prompt: str | None,
    config: PipelineConfig,
    model_override: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Generate using HuggingFace Inference API (serverless or dedicated endpoint)."""
    import httpx

    api_key = _get_api_key(config)
    model = model_override or config.llm.model

    # HuggingFace Inference API endpoint
    base_url = config.llm.base_url or "https://api-inference.huggingface.co"
    url = f"{base_url}/models/{model}/v1/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": kwargs.get("temperature", config.llm.temperature),
        "max_tokens": kwargs.get("max_tokens", config.llm.max_tokens),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = httpx.post(url, json=payload, headers=headers, timeout=120.0)
    response.raise_for_status()
    data = response.json()

    # Parse the response (HF Inference API follows OpenAI format)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})

    return LLMResponse(
        content=content,
        model=model,
        provider="huggingface",
        usage={
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    )


def _generate_ollama(
    prompt: str,
    system_prompt: str | None,
    config: PipelineConfig,
    model_override: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Generate using local Ollama instance."""
    import httpx

    model = model_override or config.llm.model
    base_url = config.llm.base_url or "http://localhost:11434"
    url = f"{base_url}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": kwargs.get("temperature", config.llm.temperature),
        },
    }

    response = httpx.post(url, json=payload, timeout=120.0)
    response.raise_for_status()
    data = response.json()

    return LLMResponse(
        content=data.get("message", {}).get("content", ""),
        model=model,
        provider="ollama",
        usage={
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        },
    )


def _generate_compatible(
    prompt: str,
    system_prompt: str | None,
    config: PipelineConfig,
    model_override: str | None = None,
    **kwargs,
) -> LLMResponse:
    """
    Generate using any OpenAI-compatible API (Groq, Together, DeepSeek, etc.)
    Just set base_url in config to point to the provider's endpoint.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=_get_api_key(config),
        base_url=config.llm.base_url,
    )
    model = model_override or config.llm.model

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=kwargs.get("temperature", config.llm.temperature),
        max_tokens=kwargs.get("max_tokens", config.llm.max_tokens),
    )

    return LLMResponse(
        content=response.choices[0].message.content or "",
        model=model,
        provider="compatible",
        usage={
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        },
    )


def _generate_local(
    prompt: str,
    system_prompt: str | None,
    config: PipelineConfig,
    model_override: str | None = None,
    **kwargs,
) -> LLMResponse:
    """
    Generate using a locally downloaded HuggingFace model via transformers.
    Downloads model on first use (~few GB depending on model), then runs on CPU/GPU.

    WARNING: Models like Mistral-7B are ~14GB and VERY slow on CPU (minutes per response).
    Recommended for smaller models only (e.g., TinyLlama, Phi-2, SmolLM).
    For 7B+ models, prefer OpenAI API — it's faster and cheaper than waiting on CPU.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model_name = model_override or config.llm.model

    # Build the full prompt
    full_prompt = ""
    if system_prompt:
        full_prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"
    else:
        full_prompt = f"[INST] {prompt} [/INST]"

    # Use transformers pipeline (handles download + caching automatically)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=kwargs.get("max_tokens", config.llm.max_tokens),
        temperature=kwargs.get("temperature", config.llm.temperature) or 0.01,
        do_sample=True,
    )

    result = pipe(full_prompt)
    generated_text = result[0]["generated_text"]

    # Remove the prompt from the output (transformers includes it)
    if generated_text.startswith(full_prompt):
        generated_text = generated_text[len(full_prompt):].strip()

    return LLMResponse(
        content=generated_text,
        model=model_name,
        provider="local",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


# =============================================================================
# Embedding Implementations
# =============================================================================


def _embed_openai(texts: list[str], config: PipelineConfig) -> EmbeddingResponse:
    """Generate embeddings using OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=_get_api_key(config))
    response = client.embeddings.create(
        model=config.embedding.model,
        input=texts,
    )
    return EmbeddingResponse(
        embeddings=[item.embedding for item in response.data],
        model=config.embedding.model,
        provider="openai",
    )


def _embed_huggingface(texts: list[str], config: PipelineConfig) -> EmbeddingResponse:
    """Generate embeddings locally using sentence-transformers (no API needed)."""
    from sentence_transformers import SentenceTransformer

    model_name = config.embedding.model

    # Cache the model — load once, reuse forever
    if not hasattr(_embed_huggingface, "_model") or _embed_huggingface._model_name != model_name:
        _embed_huggingface._model = SentenceTransformer(model_name)
        _embed_huggingface._model_name = model_name

    model = _embed_huggingface._model

    # Truncate long texts to avoid memory issues
    texts = [t[:2000] if len(t) > 2000 else t for t in texts]

    # Generate embeddings locally — no network calls
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        model=model_name,
        provider="huggingface",
    )


def _embed_ollama(texts: list[str], config: PipelineConfig) -> EmbeddingResponse:
    """Generate embeddings using Ollama."""
    import httpx

    base_url = config.llm.base_url or "http://localhost:11434"
    model = config.embedding.model
    embeddings = []

    for text in texts:
        response = httpx.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60.0,
        )
        response.raise_for_status()
        embeddings.append(response.json()["embedding"])

    return EmbeddingResponse(
        embeddings=embeddings,
        model=model,
        provider="ollama",
    )


# =============================================================================
# Public API — These are the functions you use everywhere
# =============================================================================

# Dispatch tables
_GENERATORS: dict[str, Any] = {
    "openai": _generate_openai,
    "huggingface": _generate_huggingface,
    "ollama": _generate_ollama,
    "compatible": _generate_compatible,
    "local": _generate_local,
}

_EMBEDDERS: dict[str, Any] = {
    "openai": _embed_openai,
    "huggingface": _embed_huggingface,
    "ollama": _embed_ollama,
    "compatible": _embed_openai,  # Compatible APIs often support /embeddings endpoint
}


def generate(
    prompt: str,
    system_prompt: str | None = None,
    config: PipelineConfig | None = None,
    model_override: str | None = None,
    **kwargs,
) -> LLMResponse:
    """
    Generate text using the configured LLM provider.

    Switch providers by changing config.llm.provider — no code changes needed.

    Args:
        prompt: User prompt.
        system_prompt: Optional system prompt.
        config: Pipeline config (uses global if not provided).
        model_override: Override the model name for this call only.
        **kwargs: Additional params (temperature, max_tokens).

    Returns:
        LLMResponse with content, model name, provider, and usage stats.
    """
    cfg = config or get_config()
    provider = cfg.llm.provider.lower()

    generator = _GENERATORS.get(provider)
    if not generator:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported: {list(_GENERATORS.keys())}"
        )

    return generator(prompt, system_prompt, cfg, model_override, **kwargs)


def embed_texts(
    texts: list[str],
    config: PipelineConfig | None = None,
) -> EmbeddingResponse:
    """
    Generate embeddings using the configured EMBEDDING provider.

    DECOUPLED from llm.provider — uses config.embedding.provider instead.
    This means you can use OpenAI for LLM + HuggingFace for embeddings.

    Args:
        texts: List of texts to embed.
        config: Pipeline config (uses global if not provided).

    Returns:
        EmbeddingResponse with embedding vectors.
    """
    cfg = config or get_config()
    provider = cfg.embedding.provider.lower()

    embedder = _EMBEDDERS.get(provider)
    if not embedder:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Supported: {list(_EMBEDDERS.keys())}"
        )

    return embedder(texts, cfg)


def generate_as_judge(
    prompt: str,
    system_prompt: str | None = None,
    config: PipelineConfig | None = None,
) -> LLMResponse:
    """
    Generate using the judge model (for evaluation/scoring).
    Uses config.llm.judge_model instead of config.llm.model.
    """
    cfg = config or get_config()
    return generate(
        prompt=prompt,
        system_prompt=system_prompt,
        config=cfg,
        model_override=cfg.llm.judge_model,
    )