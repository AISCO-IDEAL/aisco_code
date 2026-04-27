"""Audit GAOT3 trainer files against the mechanical Plan3 cleanup rules."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = PROJECT_ROOT / "main.py"
TRAINER_DIR = PROJECT_ROOT / "src" / "trainer"

SMALL_FILE_EXCEPTIONS = {
    "checkpoint_handler.py": "Checkpoint loading and saving is user-visible and distinct from orchestration.",
    "config_handler.py": "Config parsing is used directly by main.py and must stay isolated.",
    "edge_handler.py": "Edge precomputation would bloat Data_handler.py, which is near the 250-line target.",
    "loss_plotter.py": "Loss plot output is user-visible and separate from optimization state.",
    "metric_calculator.py": "Plan3 explicitly assigns metric normalization and metric writes here.",
    "model_handler.py": "Model construction depends on runtime-detected data shape.",
    "path_handler.py": "Path mutation and datarow initialization are isolated setup behavior.",
    "runtime_handler.py": "Runtime and distributed setup would bloat TrainingOrchestrator.",
    "statistics_handler.py": "Dataset statistics are a distinct responsibility and may stay separate.",
}


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _module_docstring(path: Path) -> str | None:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))


def _classes(path: Path) -> list[ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in tree.body if isinstance(node, ast.ClassDef)]


def _public_methods(class_node: ast.ClassDef) -> list[str]:
    return [
        node.name
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def _has_self_config_assignment(class_node: ast.ClassDef) -> bool:
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.AnnAssign):
                target = child.target
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "config"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    return True
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "config"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        return True
    return False


def collect_audit() -> dict[str, Any]:  # noqa: PLR0912
    trainer_files = sorted(path for path in TRAINER_DIR.glob("*.py") if path.name != "__init__.py")
    file_rows = []
    failures = []

    for path in trainer_files:
        classes = _classes(path)
        public_methods = {class_node.name: _public_methods(class_node) for class_node in classes}
        lines = _line_count(path)
        row = {
            "file": path.relative_to(PROJECT_ROOT).as_posix(),
            "lines": lines,
            "classes": [class_node.name for class_node in classes],
            "public_methods": public_methods,
            "small_file_exception": SMALL_FILE_EXCEPTIONS.get(path.name),
        }
        file_rows.append(row)
        if lines >= 250:
            failures.append(f"{path.name} has {lines} lines, not below the 250-line target")
        if len(classes) > 1:
            failures.append(f"{path.name} has more than one class")
        if not _module_docstring(path):
            failures.append(f"{path.name} is missing a module docstring")
        if lines < 100 and path.name not in SMALL_FILE_EXCEPTIONS:
            failures.append(f"{path.name} is under 100 lines without a documented exception")
        for class_node in classes:
            if not _has_self_config_assignment(class_node):
                failures.append(f"{path.name}:{class_node.name} does not assign self.config in __init__")  # noqa: PERF401

    main_text = MAIN_FILE.read_text(encoding="utf-8")
    if not _module_docstring(MAIN_FILE):
        failures.append("main.py is missing a module docstring")
    if main_text.count("main_config") < 2:
        failures.append("main.py does not expose and use main_config")
    if (TRAINER_DIR / "prediction_writer.py").exists():
        failures.append("prediction_writer.py still exists")
    if (TRAINER_DIR / "result_writer.py").exists():
        failures.append("result_writer.py still exists")

    loss_text = (TRAINER_DIR / "loss_fn.py").read_text(encoding="utf-8")
    for helper_name in ("build_loss_fn", "RelativeL2Loss", "L2Loss"):
        if f"def {helper_name}" in loss_text:
            failures.append(f"loss_fn.py still exposes module helper {helper_name}")  # noqa: PERF401

    return {
        "project_root": PROJECT_ROOT.as_posix(),
        "trainer_files": file_rows,
        "small_file_exceptions": SMALL_FILE_EXCEPTIONS,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GAOT3 Plan3 mechanical cleanup rules.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write the audit JSON.")
    args = parser.parse_args()

    audit = collect_audit()
    output = json.dumps(audit, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")
    print(output)
    if audit["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
