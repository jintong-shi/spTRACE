"""Minimal TOML read/write helpers.

Reading uses the standard library :mod:`tomllib` on Python 3.11+, falling back
to the ``tomli`` backport on 3.10. Writing is a small emitter covering the
subset of TOML this project's design files use (tables, sub-tables, and scalar
or homogeneous-list values), which avoids depending on a full TOML writer.

This module is deliberately *not* named ``toml``: an earlier version of the
project shipped a top-level ``toml.py`` that shadowed the PyPI ``toml``
package for anything run from the repository root.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib as _toml_reader
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as _toml_reader
    except ModuleNotFoundError as exc:  # pragma: no cover - install-time problem
        raise ModuleNotFoundError(
            "No TOML parser available. Install `tomli` (Python 3.10) or use Python 3.11+."
        ) from exc


def load(fp_or_path: Any) -> dict:
    """Load TOML from an open file object or a filesystem path."""
    if hasattr(fp_or_path, "read"):
        data = fp_or_path.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return _toml_reader.loads(data)

    with Path(fp_or_path).open("rb") as fh:
        return _toml_reader.load(fh)


def loads(text: str) -> dict:
    """Parse TOML from a string."""
    return _toml_reader.loads(text)


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Python's float repr is valid TOML and round-trips exactly.
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_format_value(x) for x in v) + "]"
    raise TypeError(
        f"Cannot serialise {type(v).__name__} to TOML. Supported types: "
        "bool, int, float, str, list/tuple of those, and dict."
    )


def _write_table(name: str, d: dict, out: list[str]) -> None:
    scalars = [(k, v) for k, v in d.items() if not isinstance(v, dict)]
    subtables = [(k, v) for k, v in d.items() if isinstance(v, dict)]

    if name:
        out.append(f"[{name}]")
    for k, v in scalars:
        out.append(f"{k} = {_format_value(v)}")
    if scalars and subtables:
        out.append("")

    for i, (k, sub) in enumerate(subtables):
        child = f"{name}.{k}" if name else k
        _write_table(child, sub, out)
        if i != len(subtables) - 1:
            out.append("")


def dumps(data: dict) -> str:
    """Serialise a nested dict to a TOML string."""
    if not isinstance(data, dict):
        raise TypeError("dump data must be a dict")
    lines: list[str] = []
    _write_table("", data, lines)
    return "\n".join(lines).rstrip() + "\n"


def dump(data: dict, fp: io.TextIOBase) -> None:
    """Write a nested dict as TOML to a text file-like object."""
    fp.write(dumps(data))
