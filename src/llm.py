"""Ollama client and model selection.

Model choice lives in config.MODELS. Switch with the HELIX_MODEL environment
variable or the --model flag on scripts/ask.py; nothing else in the codebase
needs to know which model is in use.
"""

from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _client():
    import ollama

    return ollama.Client(timeout=config.LLM_TIMEOUT_S)


def resolve_model(key: str | None = None) -> str:
    """Map a MODELS key to its Ollama tag. Falls back to config.ACTIVE_MODEL."""
    key = key or config.ACTIVE_MODEL
    if key in config.MODELS:
        return config.MODELS[key]
    # Allow passing a raw Ollama tag straight through.
    if ":" in key:
        return key
    raise ValueError(
        f"Unknown model {key!r}. Available: {', '.join(sorted(config.MODELS))}"
    )


def available_models() -> list[str]:
    """Model tags Ollama currently has locally. Empty if Ollama is unreachable."""
    try:
        response = _client().list()
    except Exception:
        return []

    tags = []
    for model in response.get("models", []):
        tag = model.get("model") or model.get("name")
        if tag:
            tags.append(tag)
    return tags


def check_ready(model_key: str | None = None) -> tuple[bool, str]:
    """Verify Ollama is running and the selected model is usable.

    Cloud models (":cloud" suffix) are not listed locally, so their absence from
    the local list is not an error.
    """
    tag = resolve_model(model_key)
    tags = available_models()

    if not tags:
        return False, (
            "Cannot reach Ollama. Is it running? Try: ollama serve"
        )
    if tag.endswith(":cloud"):
        return True, f"{tag} (cloud)"
    if tag not in tags:
        return False, (
            f"Model {tag!r} not found locally. Try: ollama pull {tag}\n"
            f"Available: {', '.join(tags)}"
        )
    return True, tag


def generate(
    prompt: str,
    system: str | None = None,
    model_key: str | None = None,
    temperature: float | None = None,
) -> str:
    """Single-turn completion. Returns the assistant's message content."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client().chat(
        model=resolve_model(model_key),
        messages=messages,
        options={
            "temperature": (
                config.LLM_TEMPERATURE if temperature is None else temperature
            ),
            "num_ctx": config.LLM_NUM_CTX,
        },
    )
    return response["message"]["content"].strip()


def stream(
    prompt: str,
    system: str | None = None,
    model_key: str | None = None,
    temperature: float | None = None,
):
    """Yield response chunks as they arrive."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for part in _client().chat(
        model=resolve_model(model_key),
        messages=messages,
        stream=True,
        options={
            "temperature": (
                config.LLM_TEMPERATURE if temperature is None else temperature
            ),
            "num_ctx": config.LLM_NUM_CTX,
        },
    ):
        piece = part.get("message", {}).get("content", "")
        if piece:
            yield piece
