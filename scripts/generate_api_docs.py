#!/usr/bin/env python3
"""Generate a small Markdown API reference from Python docstrings."""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Iterable


DEFAULT_MODULES = [
    "silver_truth.metrics.evaluation_logic",
    "silver_truth.metrics.metrics",
    "silver_truth.fusion.crops_experiment",
]


def _first_doc_line(obj: object) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def _iter_module_functions(module_name: str) -> Iterable[tuple[str, str, str]]:
    module = importlib.import_module(module_name)
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ != module.__name__:
            continue
        signature = str(inspect.signature(func))
        summary = _first_doc_line(func)
        yield name, signature, summary


def generate_markdown(modules: list[str]) -> str:
    lines = [
        "# API Reference (Generated)",
        "",
        "This page is generated from function signatures and docstrings.",
        "",
    ]
    for module_name in modules:
        lines.append(f"## `{module_name}`")
        lines.append("")
        rows = list(_iter_module_functions(module_name))
        if not rows:
            lines.append("_No functions found._")
            lines.append("")
            continue

        lines.append("| Function | Signature | Summary |")
        lines.append("|---|---|---|")
        for func_name, signature, summary in rows:
            escaped_summary = summary.replace("|", "\\|")
            lines.append(f"| `{func_name}` | `{signature}` | {escaped_summary} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/api_reference_generated.md"),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=DEFAULT_MODULES,
        help="Modules to document.",
    )
    args = parser.parse_args()

    markdown = generate_markdown(args.modules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
