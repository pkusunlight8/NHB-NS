from __future__ import annotations
from typing import Any, Iterable

class MissingInputError(RuntimeError):
    """A scientifically necessary input is absent or has not been verified."""

    def __init__(self, message: str, required: Iterable[str] | None = None):
        self.required = list(required or [])
        super().__init__(message)


class DataContractError(ValueError):
    """An input exists but violates its declared grain, values, or units."""


def require_columns(frame: Any, columns: Iterable[str], context: str = "table") -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MissingInputError(f"{context}: missing columns: {', '.join(missing)}", missing)


def require_unique(frame: Any, keys: list[str], context: str = "table") -> None:
    require_columns(frame, keys, context)
    if frame[keys].isna().any().any():
        raise DataContractError(f"{context}: null key in {keys}")
    n = int(frame.duplicated(keys, keep=False).sum())
    if n:
        raise DataContractError(f"{context}: {n} rows have duplicated keys {keys}")
