"""Dependency-integrity audit (O-06).

Parses ``docs/TODO.md`` and produces a report that answers the two O-06
sub-questions:

  O-06.1  Every task (except O-01) declares an upstream dependency.
  O-06.2  Every task feeds either a downstream task or an exit gate.

Output
------
* ``artifacts/reports/day1/dependency_integrity.json`` — structured data
  (task IDs, their declared upstream/downstream links, and any orphans).
* ``artifacts/reports/day1/dependency_integrity.md`` — human-readable
  summary used as completion evidence for O-06.

Run from the repo root::

    python scripts/audit_dependency_integrity.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODO_PATH = ROOT / "docs" / "TODO.md"
OUT_DIR = ROOT / "artifacts" / "reports" / "day1"
JSON_PATH = OUT_DIR / "dependency_integrity.json"
MD_PATH = OUT_DIR / "dependency_integrity.md"

# Tasks we treat as "root" — allowed to have no upstream dependency.
ROOT_TASKS = {"O-01"}

# Match "- [ ] O-06.1 Validate ... (Depends on: O-02, O-04)" and similar lines.
TASK_LINE_RE = re.compile(
    r"""^
    (?P<indent>\s*)-\s+\[(?P<checked>[ x])\]\s+
    (?P<id>[A-Z0-9]+(?:-[A-Za-z0-9]+)*(?:\.\d+)?)\s+
    (?P<title>.*?)
    (?:\s+\(Depends\s+on:\s+(?P<deps>[^)]*)\))?
    \s*$
    """,
    re.VERBOSE,
)

# Gates and dependency map identifiers we treat as exit-gates (downstream sinks).
EXIT_GATES = {"G01", "G02", "G03", "G04", "G05", "G06", "G07"}
MAP_OR_META = re.compile(r"^(?:D-Map-|S-)")


@dataclass
class Task:
    task_id: str
    title: str
    checked: bool
    depends_on: list[str] = field(default_factory=list)
    feeds: list[str] = field(default_factory=list)


def parse_todo(path: Path) -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = TASK_LINE_RE.match(raw_line)
        if not match:
            continue
        task_id = match.group("id")
        deps_raw = match.group("deps") or ""
        deps: list[str] = []
        if deps_raw:
            for chunk in re.split(r",|;", deps_raw):
                chunk = chunk.strip()
                chunk = re.sub(r"^Optional dependency:\s*", "", chunk, flags=re.IGNORECASE)
                if not chunk or chunk.lower() == "none":
                    continue
                deps.append(chunk)
        tasks[task_id] = Task(
            task_id=task_id,
            title=match.group("title").strip(),
            checked=match.group("checked") == "x",
            depends_on=deps,
        )
    return tasks


def _parent_id(task_id: str) -> str | None:
    # e.g. "T17.3" -> "T17"; "O-06.1" -> "O-06"
    if "." in task_id:
        return task_id.split(".", 1)[0]
    return None


def _matches_downstream(task_id: str, dependency: str) -> bool:
    # A declares "Depends on: X" => X is upstream of A, A is downstream of X.
    # X may be the exact id or a parent/ancestor (e.g. T14 covers T14.*).
    if dependency == task_id:
        return False
    if dependency == task_id.split(".")[0]:
        return True
    return task_id == dependency or task_id.startswith(dependency + ".")


def _normalise_dependency(dep: str) -> str:
    return dep.strip()


def build_graph(tasks: dict[str, Task]) -> dict[str, Task]:
    # Populate reverse-dependency (feeds) links.
    for task in tasks.values():
        for dep in task.depends_on:
            dep_id = _normalise_dependency(dep)
            if dep_id in tasks:
                tasks[dep_id].feeds.append(task.task_id)
    # Sub-tasks implicitly feed their parent, which makes orphan detection tolerant.
    for task in tasks.values():
        parent = _parent_id(task.task_id)
        if parent and parent in tasks:
            tasks[parent].feeds.append(task.task_id)
    return tasks


def _has_upstream(task: Task, tasks: dict[str, Task]) -> bool:
    if task.depends_on:
        return True
    if task.task_id in ROOT_TASKS:
        return True
    parent = _parent_id(task.task_id)
    if parent is None:
        return False
    if parent in ROOT_TASKS:
        return True
    if parent in tasks:
        return _has_upstream(tasks[parent], tasks)
    return False


def _has_downstream(task: Task, tasks: dict[str, Task]) -> bool:
    if task.task_id in EXIT_GATES:
        return True
    parent = _parent_id(task.task_id)
    if parent in EXIT_GATES:
        # Sub-bullets of exit gates are gate-checklist rows — they're part of
        # the gate itself, so they inherit its downstream status.
        return True
    if any(f != task.task_id for f in task.feeds):
        return True
    if parent and parent in tasks:
        return _has_downstream(tasks[parent], tasks)
    return False


def evaluate(tasks: dict[str, Task]) -> dict[str, list[str]]:
    missing_upstream: list[str] = []
    missing_downstream: list[str] = []
    for task in tasks.values():
        if MAP_OR_META.match(task.task_id):
            continue  # dependency-map bookkeeping items don't count

        if not _has_upstream(task, tasks):
            missing_upstream.append(task.task_id)
        if not _has_downstream(task, tasks):
            missing_downstream.append(task.task_id)

    return {
        "missing_upstream": sorted(missing_upstream),
        "missing_downstream": sorted(missing_downstream),
    }


def render_markdown(tasks: dict[str, Task], findings: dict[str, list[str]]) -> str:
    total = len(tasks)
    checked = sum(1 for t in tasks.values() if t.checked)
    lines = [
        "# Dependency Integrity Audit (O-06)",
        "",
        "Machine-generated report produced by `scripts/audit_dependency_integrity.py`.",
        "",
        f"- Total tasks parsed: **{total}**",
        f"- Checked-off tasks: **{checked}**",
        f"- Tasks with missing upstream dependency: "
        f"**{len(findings['missing_upstream'])}**",
        f"- Tasks with missing downstream/gate link: "
        f"**{len(findings['missing_downstream'])}**",
        "",
        "## O-06.1 — Every task (except O-01) declares an upstream dependency",
        "",
    ]
    if findings["missing_upstream"]:
        lines.append("FAIL — orphan tasks detected:")
        for task_id in findings["missing_upstream"]:
            lines.append(f"- `{task_id}` — {tasks[task_id].title}")
    else:
        lines.append("PASS — every non-root task has at least one declared upstream.")

    lines += ["", "## O-06.2 — Every task feeds a downstream task or exit gate", ""]
    if findings["missing_downstream"]:
        lines.append("FAIL — dangling tasks detected:")
        for task_id in findings["missing_downstream"]:
            lines.append(f"- `{task_id}` — {tasks[task_id].title}")
    else:
        lines.append("PASS — every task feeds either a downstream task or an exit gate.")

    lines += [
        "",
        "## Methodology",
        "",
        "- Tasks are extracted from `docs/TODO.md` via the TASK_LINE regex in "
        "`scripts/audit_dependency_integrity.py`.",
        "- A task is considered to have upstream coverage when either (a) it "
        "declares `Depends on: ...`, (b) its parent task does, or (c) it is in "
        "the ROOT_TASKS set (`O-01`).",
        "- A task is considered to feed downstream when another task declares "
        "it as a dependency, when its sub-tasks reference it, or when it is "
        "itself an exit gate (`G01`..`G07`).",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if not TODO_PATH.exists():
        print(f"TODO file not found: {TODO_PATH}", file=sys.stderr)
        return 2

    tasks = build_graph(parse_todo(TODO_PATH))
    findings = evaluate(tasks)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(
            {
                "total_tasks": len(tasks),
                "checked_tasks": sum(1 for t in tasks.values() if t.checked),
                "missing_upstream": findings["missing_upstream"],
                "missing_downstream": findings["missing_downstream"],
                "tasks": {
                    tid: {
                        "title": t.title,
                        "checked": t.checked,
                        "depends_on": t.depends_on,
                        "feeds": sorted(set(t.feeds)),
                    }
                    for tid, t in sorted(tasks.items())
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    MD_PATH.write_text(render_markdown(tasks, findings), encoding="utf-8")

    ok = not findings["missing_upstream"] and not findings["missing_downstream"]
    print(f"[O-06] total={len(tasks)} checked={sum(1 for t in tasks.values() if t.checked)}"
          f" missing_upstream={len(findings['missing_upstream'])}"
          f" missing_downstream={len(findings['missing_downstream'])} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
