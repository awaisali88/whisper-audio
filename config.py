"""Loads environment variables from a local `.env` file at import time.

Side-effectful by design: importing this module populates `os.environ` from
`.env` without overriding values that are already set in the real environment.
Import it before any module that reads HF_* / OLLAMA_* env vars.
"""
import os
from pathlib import Path

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None


def _load() -> None:
    if dotenv_values is None:
        return
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for key, value in dotenv_values(env_path).items():
        if value and key not in os.environ:
            os.environ[key] = value


def _fix_ollama_host() -> None:
    # On Windows, clients cannot dial 0.0.0.0 (it's a bind-only address). The
    # Ollama server legitimately binds to 0.0.0.0 and users often have
    # OLLAMA_HOST=0.0.0.0:11434 set system-wide; rewrite it for client use.
    host = os.environ.get("OLLAMA_HOST", "")
    if host and "0.0.0.0" in host:
        os.environ["OLLAMA_HOST"] = host.replace("0.0.0.0", "127.0.0.1")


_load()
_fix_ollama_host()
