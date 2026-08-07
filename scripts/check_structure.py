"""Validate the repository's canonical source structure.

The checker inspects project-owned source and tracked repository content rather
than transient files created by tests, linters, virtual environments, or
Python itself. Run from the repository root with:

    python scripts/check_structure.py
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    """Print violations and return a non-zero exit code when any exist."""

    violations = [
        *check_tracked_generated_files(),
        *check_forbidden_modules(),
        *check_module_sizes(),
        *check_runtime_imports(),
    ]
    if violations:
        print("Structure violations:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Structure check passed.")
    return 0


def check_tracked_generated_files() -> list[str]:
    """Reject generated artifacts only when Git actually tracks them."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    violations: list[str] = []
    tracked = result.stdout.decode("utf-8", errors="surrogateescape")
    for raw_path in tracked.split("\0"):
        if not raw_path:
            continue
        path = Path(raw_path)
        if any(
            part in GENERATED_PARTS or part.endswith(".egg-info") for part in path.parts
        ) or path.suffix in {".pyc", ".pyo"}:
            violations.append(f"generated path is tracked: {path.as_posix()}")
    return violations


def check_forbidden_modules() -> list[str]:
    """Reject vague module names only inside project-owned Python code."""

    violations: list[str] = []
    bases = (ROOT / "src", ROOT / "research", ROOT / "scripts", ROOT / "tests")
    for base in bases:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.name in FORBIDDEN_NAMES:
                violations.append(f"forbidden generic module: {path.relative_to(ROOT)}")
    return violations


def check_module_sizes() -> list[str]:
    """Reject Python source modules above the agreed hard ceiling."""

    violations: list[str] = []
    for base in (ROOT / "src", ROOT / "research"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            lines = len(path.read_text(encoding="utf-8").splitlines())
            if lines > MAX_MODULE_LINES:
                violations.append(
                    f"{path.relative_to(ROOT)} has {lines} lines; maximum is {MAX_MODULE_LINES}"
                )
    return violations


def check_runtime_imports() -> list[str]:
    """Reject imports from the research package inside installed runtime code."""

    violations: list[str] = []
    runtime = ROOT / "src" / "pii_scrub"
    if not runtime.exists():
        return violations
    for path in runtime.rglob("*.py"):
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
