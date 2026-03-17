"""Load historical candle data for backtesting.

Supports CSV files with columns: timestamp, open, high, low, close, volume.
Timestamps are parsed according to a configurable format string.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from aiswarm.backtest.engine import OHLCV


def load_candles_from_csv(
    path: str | Path,
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> list[OHLCV]:
    """Load OHLCV candles from a CSV file.

    Expected CSV columns (case-sensitive):
        timestamp, open, high, low, close, volume

    Args:
        path: Filesystem path to the CSV file.
        date_format: ``strptime`` format for the timestamp column.

    Returns:
        Chronologically sorted list of ``OHLCV`` bars.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If a required column is missing.
        ValueError: If a numeric value cannot be parsed.
    """
    candles: list[OHLCV] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(
                OHLCV(
                    timestamp=datetime.strptime(row["timestamp"], date_format),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return sorted(candles, key=lambda c: c.timestamp)
