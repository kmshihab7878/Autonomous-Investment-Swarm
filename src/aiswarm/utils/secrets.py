"""Secrets management abstraction layer.

Provides pluggable backends for retrieving secrets without hard-coding
``os.environ.get(...)`` throughout the codebase.  The default backend
(``EnvSecretsProvider``) preserves the existing behaviour; additional
backends support JSON files, Kubernetes-style secret directories, and
chaining multiple sources with first-match-wins semantics.

Usage
-----
At bootstrap time::

    from aiswarm.utils.secrets import create_secrets_provider, set_secrets_provider
    provider = create_secrets_provider()
    set_secrets_provider(provider)

Anywhere else::

    from aiswarm.utils.secrets import get_secrets_provider
    secret = get_secrets_provider().get_secret_required("AIS_RISK_HMAC_SECRET")

The factory ``create_secrets_provider()`` inspects ``AIS_SECRETS_FILE`` and
``AIS_SECRETS_DIR`` environment variables to decide which backends to
configure.  Environment variables are **always** included as a fallback so
existing deployments continue to work unchanged.
"""

from __future__ import annotations

import json
import os
import pathlib
from abc import ABC, abstractmethod

from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SecretsProvider(ABC):
    """Abstract secrets provider."""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by key.  Returns ``None`` if not found."""
        ...

    def get_secret_required(self, key: str) -> str:
        """Retrieve a required secret; raises ``ValueError`` if missing."""
        value = self.get_secret(key)
        if value is None:
            raise ValueError(f"Required secret '{key}' not found")
        return value


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class EnvSecretsProvider(SecretsProvider):
    """Read secrets from environment variables (current behaviour, default)."""

    def get_secret(self, key: str) -> str | None:
        return os.environ.get(key)


class FileSecretsProvider(SecretsProvider):
    """Read secrets from a JSON file or a directory of files.

    Supports two layouts:

    * **Single JSON file** -- a flat ``{"KEY": "value", ...}`` object.
    * **Directory of files** -- each file name is treated as a key and its
      content (stripped of trailing whitespace) as the value.  This matches
      the Kubernetes ``/var/run/secrets/`` convention.

    Hidden files (name starting with ``"."``) inside directories are skipped.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        p = pathlib.Path(self.path)
        if p.is_file():
            with open(p) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError(
                    f"Secrets file must contain a JSON object, got {type(data).__name__}"
                )
            self._cache = {str(k): str(v) for k, v in data.items()}
            logger.info(
                "Secrets loaded from file",
                extra={"extra_json": {"path": self.path, "count": len(self._cache)}},
            )
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and not child.name.startswith("."):
                    self._cache[child.name] = child.read_text().strip()
            logger.info(
                "Secrets loaded from directory",
                extra={"extra_json": {"path": self.path, "count": len(self._cache)}},
            )
        else:
            raise FileNotFoundError(f"Secrets path not found: {self.path}")

    def get_secret(self, key: str) -> str | None:
        return self._cache.get(key)


class ChainSecretsProvider(SecretsProvider):
    """Chain multiple providers; first non-``None`` result wins."""

    def __init__(self, providers: list[SecretsProvider]) -> None:
        if not providers:
            raise ValueError("ChainSecretsProvider requires at least one provider")
        self.providers = list(providers)

    def get_secret(self, key: str) -> str | None:
        for provider in self.providers:
            value = provider.get_secret(key)
            if value is not None:
                return value
        return None


# ---------------------------------------------------------------------------
# Module-level default provider (singleton pattern)
# ---------------------------------------------------------------------------

_default_provider: SecretsProvider | None = None


def get_secrets_provider() -> SecretsProvider:
    """Return the module-level secrets provider.

    If ``set_secrets_provider()`` has not been called, returns a plain
    ``EnvSecretsProvider`` so that existing call-sites work without any
    bootstrap step (backward compatible).
    """
    global _default_provider  # noqa: PLW0603
    if _default_provider is None:
        _default_provider = EnvSecretsProvider()
    return _default_provider


def set_secrets_provider(provider: SecretsProvider) -> None:
    """Set the module-level secrets provider.

    Should be called once during application startup (bootstrap).
    """
    global _default_provider  # noqa: PLW0603
    _default_provider = provider
    logger.info(
        "Secrets provider configured",
        extra={"extra_json": {"type": type(provider).__name__}},
    )


def reset_secrets_provider() -> None:
    """Reset to the default provider.  Intended for testing only."""
    global _default_provider  # noqa: PLW0603
    _default_provider = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_secrets_provider() -> SecretsProvider:
    """Create the appropriate secrets provider based on environment.

    Configuration via environment variables:

    * ``AIS_SECRETS_FILE=/path/to/secrets.json`` -- prepend a
      ``FileSecretsProvider`` reading from a JSON file.
    * ``AIS_SECRETS_DIR=/path/to/secrets/`` -- prepend a
      ``FileSecretsProvider`` reading from a directory of files.
    * If neither is set, return a plain ``EnvSecretsProvider``.

    When file/directory providers are configured, ``EnvSecretsProvider``
    is **always** appended as a fallback so environment-variable overrides
    continue to work.
    """
    providers: list[SecretsProvider] = []

    secrets_file = os.environ.get("AIS_SECRETS_FILE")
    secrets_dir = os.environ.get("AIS_SECRETS_DIR")

    if secrets_file:
        providers.append(FileSecretsProvider(secrets_file))
        logger.info(
            "Secrets provider added",
            extra={"extra_json": {"type": "file", "path": secrets_file}},
        )

    if secrets_dir:
        providers.append(FileSecretsProvider(secrets_dir))
        logger.info(
            "Secrets provider added",
            extra={"extra_json": {"type": "directory", "path": secrets_dir}},
        )

    # Always include environment variables as the final fallback
    providers.append(EnvSecretsProvider())

    if len(providers) == 1:
        return providers[0]
    return ChainSecretsProvider(providers)
