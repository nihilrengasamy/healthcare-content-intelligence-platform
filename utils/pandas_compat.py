"""Small table-data compatibility layer for restricted Windows environments.

This project normally uses Pandas. Some locked-down Windows machines block
compiled Pandas extension modules from virtual environments. The Streamlit UI
can still render demo tables and charts with simple row dictionaries, so this
module provides the tiny subset of DataFrame behavior used by the frontend.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


class SimpleColumn(list[Any]):
    """List-like column with the small API used by chart pages."""

    def tolist(self) -> list[Any]:
        """Return column values as a list."""
        return list(self)


class SimpleFrame(list[dict[str, Any]]):
    """List-backed table with minimal DataFrame-like helpers."""

    @property
    def empty(self) -> bool:
        """Return whether the table has rows."""
        return len(self) == 0

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        """Return rows as dictionaries.

        Args:
            orient: Supported orientation. Only ``records`` is meaningful.

        Returns:
            Row dictionaries.
        """
        return [dict(row) for row in self]

    def drop(
        self,
        columns: list[str] | None = None,
        errors: str = "raise",
    ) -> "SimpleFrame":
        """Return a table without selected columns.

        Args:
            columns: Column names to remove.
            errors: Included for Pandas API compatibility.

        Returns:
            New simplified frame.
        """
        del errors
        blocked = set(columns or [])
        return SimpleFrame([{key: value for key, value in row.items() if key not in blocked} for row in self])

    def __getitem__(self, key: str) -> SimpleColumn:
        """Return a list-like column by name."""
        return SimpleColumn([row.get(key) for row in self])


class SimpleSeries(list[Any]):
    """List-backed series with value counting support."""

    def value_counts(self) -> "ValueCounts":
        """Count values in the series."""
        return ValueCounts(Counter(self))


class ValueCounts:
    """Minimal value-count result that can reset to a table."""

    def __init__(self, counts: Counter[Any]) -> None:
        """Initialize with counted values."""
        self._counts = counts
        self.columns: list[str] = ["index", "count"]

    def reset_index(self) -> SimpleFrame:
        """Return value counts as rows."""
        label, count_label = self.columns
        return SimpleFrame([{label: key, count_label: value} for key, value in self._counts.items()])


def DataFrame(data: Any = None) -> SimpleFrame:
    """Create a simple table from common row-oriented inputs."""
    if data is None:
        return SimpleFrame()
    if isinstance(data, SimpleFrame):
        return data
    if isinstance(data, Mapping):
        return SimpleFrame([dict(data)])
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return SimpleFrame([dict(row) for row in data if isinstance(row, Mapping)])
    return SimpleFrame()


def Series(values: Iterable[Any]) -> SimpleSeries:
    """Create a simple series."""
    return SimpleSeries(values)
