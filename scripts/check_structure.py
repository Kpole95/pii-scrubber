"""Validate repository structure, module size, and documentation rules."""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "research", "scripts", "tests")
FORBIDDEN_NAMES = {"utils.py", "helpers.py", "manager.py", "processor.py"}
GENERATED_PARTS = {
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
}
MAX_MODULE_LINES = 300


def main() -> int:
    """Print structure violations and return a process exit code."""

    violations = [
        *check_tracked_generated_files(),
        *check_forbidden_modules(),
        *check_module_sizes(),
        *check_docstrings(),
        *check_runtime_imports(),
    ]
    if not violations:
        print("Structure check passed.")
        return 0

    print("Structure violations:")
    for violation in violations:
        print(f"- {violation}")
    return 1


def project_modules() -> Iterator[Path]:
    """Yield project-owned Python modules in stable path order."""

    for directory in SOURCE_DIRS:
        base = ROOT / directory
        if base.exists():
            yield from sorted(base.rglob("*.py"))


def check_tracked_generated_files() -> list[str]:
    """Reject generated artifacts when Git tracks them."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    tracked = result.stdout.decode("utf-8", errors="surrogateescape")
    violations: list[str] = []
    for raw_path in tracked.split("\0"):
        if not raw_path:
            continue
        path = Path(raw_path)
        generated = any(
            part in GENERATED_PARTS or part.endswith(".egg-info") for part in path.parts
        )
        if generated or path.suffix in {".pyc", ".pyo"}:
            violations.append(f"generated path is tracked: {path.as_posix()}")
    return violations


def check_forbidden_modules() -> list[str]:
    """Reject vague module names in project-owned Python code."""

    return [
        f"forbidden generic module: {path.relative_to(ROOT)}"
        for path in project_modules()
        if path.name in FORBIDDEN_NAMES
    ]


def check_module_sizes() -> list[str]:
    """Reject project modules above the agreed 300-line ceiling."""

    violations: list[str] = []
    for path in project_modules():
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > MAX_MODULE_LINES:
            violations.append(
                f"{path.relative_to(ROOT)} has {lines} lines; maximum is {MAX_MODULE_LINES}"
            )
    return violations


def check_docstrings() -> list[str]:
    """Require short documentation on modules, classes, and functions."""

    violations: list[str] = []
    documented_nodes = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for path in project_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT)
        if ast.get_docstring(tree) is None:
            violations.append(f"missing module docstring: {relative}")

        for node in ast.walk(tree):
            if isinstance(node, documented_nodes) and ast.get_docstring(node) is None:
                violations.append(f"missing docstring: {relative}:{node.lineno} {node.name}")
    return violations


def check_runtime_imports() -> list[str]:
    """Reject research imports from the installed runtime package."""

    violations: list[str] = []
    runtime = ROOT / "src" / "pii_scrub"
    for path in sorted(runtime.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "research" or name.startswith("research.") for name in names):
                violations.append(f"runtime imports research: {path.relative_to(ROOT)}")
    return violations


if __name__ == "__main__":
    raise SystemExit(main())
