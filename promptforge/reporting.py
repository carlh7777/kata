from __future__ import annotations

import fnmatch
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathPolicyResult:
    passed: bool
    allowed_rules: list[str]
    forbidden_rules: list[str]
    violating_paths: list[str]


def render_report(run_ref: str) -> str:
    run_root = resolve_run_root(run_ref)
    summary = load_summary(run_root / "run_summary.json")
    task_reports = [build_task_report(run_root, task) for task in summary["tasks"]]

    promptforge_wins = sum(1 for task in task_reports if task["outcome"] == "PromptForge win")
    baseline_wins = sum(1 for task in task_reports if task["outcome"] == "Baseline win")
    ties = sum(1 for task in task_reports if task["outcome"] == "Tie")
    invalid = sum(1 for task in task_reports if task["outcome"] == "Invalid run")
    comparable = promptforge_wins + baseline_wins + ties

    lines: list[str] = []
    lines.append(f"# PromptForge Eval Report: {summary['run_id']}")
    lines.append("")
    lines.append(f"- Created: {summary['created_at']}")
    lines.append(f"- Run kind: {summary.get('run_kind', 'eval')}")
    lines.append(f"- Mode: {summary['mode']}")
    lines.append(f"- Requested repo: `{summary['requested_repo_ref']}`")
    lines.append(f"- Eval pack: `{summary['eval_pack']}`")
    lines.append(f"- Agent command: `{summary['agent_command']}`")
    metadata = summary.get("metadata") or {}
    if metadata.get("evaluator_version"):
        lines.append(f"- Evaluator version: `{metadata['evaluator_version']}`")
    if metadata.get("task_pool_fingerprint"):
        lines.append(f"- Task pool fingerprint: `{metadata['task_pool_fingerprint']}`")
    lines.append("")
    lines.append("## Measurement Basis")
    lines.append("- Task solved is measured by the task `checks.sh` exit status.")
    lines.append(
        "- Protected or forbidden path compliance is measured from the actual changed files "
        "against `allowed_paths.txt` and `forbidden_paths.txt`."
    )
    lines.append(
        "- Repo-rule compliance and scoring or review misunderstandings are only reported "
        "when explicitly encoded by the task checks. They are otherwise marked as not "
        "separately measured."
    )
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append(f"- Tasks: {len(task_reports)}")
    lines.append(f"- PromptForge wins: {promptforge_wins}")
    lines.append(f"- Baseline wins: {baseline_wins}")
    lines.append(f"- Ties: {ties}")
    lines.append(f"- Invalid runs: {invalid}")
    if comparable:
        lines.append(f"- PromptForge win rate: {promptforge_wins}/{comparable}")
    else:
        lines.append(
            "- PromptForge win rate: not available because no task produced a comparable "
            "result."
        )

    for task_report in task_reports:
        lines.append("")
        lines.append(f"## Task: {task_report['task_id']}")
        lines.append(f"- Repo ref: `{task_report['task_repo_ref']}`")
        lines.append(f"- Outcome: {task_report['outcome']}")
        lines.append("")
        lines.append("| Metric | Baseline | Generated |")
        lines.append("| --- | --- | --- |")
        lines.append(
            "| Agent command | "
            f"{render_status(task_report['baseline']['agent_ok'])} | "
            f"{render_status(task_report['generated']['agent_ok'])} |"
        )
        lines.append(
            "| Task solved | "
            f"{render_status(task_report['baseline']['task_solved'])} | "
            f"{render_status(task_report['generated']['task_solved'])} |"
        )
        lines.append(
            "| Checks passed | "
            f"{render_status(task_report['baseline']['checks_passed'])} | "
            f"{render_status(task_report['generated']['checks_passed'])} |"
        )
        lines.append(
            "| Protected/forbidden paths avoided | "
            f"{render_status(task_report['baseline']['path_policy_passed'])} | "
            f"{render_status(task_report['generated']['path_policy_passed'])} |"
        )
        lines.append(
            "| Repo rules followed | "
            f"{task_report['baseline']['repo_rules_followed']} | "
            f"{task_report['generated']['repo_rules_followed']} |"
        )
        lines.append(
            "| Scoring/review misunderstanding detected | "
            f"{task_report['baseline']['scoring_review_misunderstanding']} | "
            f"{task_report['generated']['scoring_review_misunderstanding']} |"
        )
        lines.append("")
        lines.append(
            "- Baseline changed paths: "
            f"{render_paths(task_report['baseline']['changed_paths'])}"
        )
        lines.append(
            "- Baseline path issues: "
            f"{render_paths(task_report['baseline']['path_policy_violations'])}"
        )
        lines.append(
            "- Generated changed paths: "
            f"{render_paths(task_report['generated']['changed_paths'])}"
        )
        lines.append(
            "- Generated path issues: "
            f"{render_paths(task_report['generated']['path_policy_violations'])}"
        )

    return "\n".join(lines)


def resolve_run_root(run_ref: str) -> Path:
    candidate = Path(run_ref).expanduser()
    if candidate.is_file() and candidate.name == "run_summary.json":
        return candidate.parent.resolve()
    if candidate.is_dir():
        return candidate.resolve()

    default_root = Path("runs") / run_ref
    if default_root.is_dir():
        return default_root.resolve()
    raise FileNotFoundError(f"Run artifacts not found: {run_ref}")


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_task_report(run_root: Path, task: dict[str, Any]) -> dict[str, Any]:
    task_root = run_root / "tasks" / task["task_id"]
    repo_snapshot = task_root / "repo_snapshot"
    eval_pack_root = task_root / "eval_pack"
    baseline = build_variant_report(repo_snapshot, eval_pack_root, task["variants"][0])
    generated = build_variant_report(repo_snapshot, eval_pack_root, task["variants"][1])
    return {
        "task_id": task["task_id"],
        "task_repo_ref": task["task_repo_ref"],
        "baseline": baseline,
        "generated": generated,
        "outcome": compare_variants(baseline, generated),
    }


def build_variant_report(
    repo_snapshot: Path,
    eval_pack_root: Path,
    variant: dict[str, Any],
) -> dict[str, Any]:
    changed_paths = diff_paths(repo_snapshot, Path(variant["workspace"]))
    path_policy = evaluate_path_policy(
        changed_paths,
        allowed_rules=read_path_rules(eval_pack_root / "allowed_paths.txt"),
        forbidden_rules=read_path_rules(eval_pack_root / "forbidden_paths.txt"),
    )
    checks_passed = variant["checks_exit_code"] == 0
    return {
        "agent_ok": variant["agent_exit_code"] == 0,
        "task_solved": checks_passed,
        "checks_passed": checks_passed,
        "path_policy_passed": path_policy.passed,
        "path_policy_violations": path_policy.violating_paths,
        "changed_paths": changed_paths,
        "repo_rules_followed": "not separately measured",
        "scoring_review_misunderstanding": "not separately measured",
        "success_score": score_variant(
            agent_ok=variant["agent_exit_code"] == 0,
            checks_passed=checks_passed,
            path_policy_passed=path_policy.passed,
        ),
    }


def diff_paths(source: Path, target: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--no-index", "--name-status", str(source), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "Unable to diff eval workspaces.")

    source_prefixes = path_prefixes(source)
    target_prefixes = path_prefixes(target)
    paths: list[str] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        status, _, raw_path = stripped.partition("\t")
        if not raw_path:
            continue
        prefixes = source_prefixes if status.startswith(("D", "M", "R", "C")) else target_prefixes
        normalized = normalize_diff_path(raw_path, prefixes)
        paths.append(normalized)
    return sorted(set(paths))


def path_prefixes(root: Path) -> list[str]:
    return [
        root.as_posix().rstrip("/") + "/",
        root.resolve().as_posix().rstrip("/") + "/",
    ]


def normalize_diff_path(path: str, prefixes: list[str]) -> str:
    normalized = path.replace("\\", "/")
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def read_path_rules(path: Path) -> list[str]:
    rules: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rules.append(stripped.replace("\\", "/").strip("/"))
    return rules


def evaluate_path_policy(
    changed_paths: list[str],
    *,
    allowed_rules: list[str],
    forbidden_rules: list[str],
) -> PathPolicyResult:
    violations: list[str] = []
    for changed_path in changed_paths:
        if matches_any_rule(changed_path, forbidden_rules):
            violations.append(changed_path)
            continue
        if allowed_rules and not matches_any_rule(changed_path, allowed_rules):
            violations.append(changed_path)
    return PathPolicyResult(
        passed=not violations,
        allowed_rules=allowed_rules,
        forbidden_rules=forbidden_rules,
        violating_paths=violations,
    )


def matches_any_rule(path: str, rules: list[str]) -> bool:
    if not rules:
        return False
    normalized = path.replace("\\", "/").strip("/")
    for rule in rules:
        if fnmatch.fnmatch(normalized, rule):
            return True
        if normalized == rule or normalized.startswith(f"{rule}/"):
            return True
    return False


def compare_variants(baseline: dict[str, Any], generated: dict[str, Any]) -> str:
    baseline_success = variant_completed_task(baseline)
    generated_success = variant_completed_task(generated)
    if generated_success and not baseline_success:
        return "PromptForge win"
    if baseline_success and not generated_success:
        return "Baseline win"
    if not baseline["agent_ok"] and not generated["agent_ok"]:
        return "Invalid run"
    return "Tie"


def score_variant(*, agent_ok: bool, checks_passed: bool, path_policy_passed: bool) -> int:
    return int(agent_ok and checks_passed and path_policy_passed)


def variant_completed_task(variant: dict[str, Any]) -> bool:
    return bool(variant["agent_ok"] and variant["task_solved"] and variant["path_policy_passed"])


def render_paths(paths: list[str]) -> str:
    return ", ".join(f"`{path}`" for path in paths) if paths else "none"


def render_status(value: bool) -> str:
    return "pass" if value else "fail"
