from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiswarm.utils.secrets import SecretsProvider


class Venue(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"


@dataclass(frozen=True)
class AsterConfig:
    """Configuration for Aster DEX connectivity."""

    account_id: str = ""
    default_venue: Venue = Venue.FUTURES
    rate_limit_calls_per_second: float = 5.0
    request_timeout_seconds: int = 10
    max_retries: int = 3

    @classmethod
    def from_env(cls, secrets_provider: SecretsProvider | None = None) -> AsterConfig:
        """Build config from environment or a secrets provider.

        When ``secrets_provider`` is ``None``, falls back to reading
        ``os.environ`` directly (preserving backward compatibility).
        """
        if secrets_provider is not None:
            account_id = secrets_provider.get_secret("ASTER_ACCOUNT_ID") or ""
        else:
            account_id = os.environ.get("ASTER_ACCOUNT_ID", "")
        return cls(
            account_id=account_id,
        )

    @property
    def has_account(self) -> bool:
        return bool(self.account_id)


# Canonical symbol mapping
_SYMBOL_MAP: dict[str, str] = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "BNB/USDT": "BNBUSDT",
    "ARB/USDT": "ARBUSDT",
    "AVAX/USDT": "AVAXUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "MATIC/USDT": "MATICUSDT",
}

_REVERSE_MAP: dict[str, str] = {v: k for k, v in _SYMBOL_MAP.items()}


def normalize_symbol(symbol: str) -> str:
    """Convert any symbol format to Aster DEX format (e.g. BTC/USDT -> BTCUSDT)."""
    if symbol in _SYMBOL_MAP:
        return _SYMBOL_MAP[symbol]
    # Already in BTCUSDT format or unknown
    return symbol.replace("/", "").replace("-", "").upper()


def to_canonical_symbol(aster_symbol: str) -> str:
    """Convert Aster DEX symbol to canonical format (e.g. BTCUSDT -> BTC/USDT)."""
    if aster_symbol in _REVERSE_MAP:
        return _REVERSE_MAP[aster_symbol]
    # Try to split USDT pairs
    if aster_symbol.endswith("USDT"):
        base = aster_symbol[:-4]
        return f"{base}/USDT"
    return aster_symbol


WHITELISTED_SYMBOLS: list[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]
