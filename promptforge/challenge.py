from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from promptforge.eval_runner import EvalRunSummary, run_prompt_variants
from promptforge.frontier import (
    FrontierManifest,
    FrontierModeConfig,
    load_frontier_manifest,
)
from promptforge.provenance import EVALUATOR_VERSION, sha256_text, short_hash


@dataclass(frozen=True)
class ChallengePoolSummary:
    task_ids: list[str]
    eval_run_summary: str
    variant_successes: dict[str, int]
    candidate_beats_frontier: bool


@dataclass(frozen=True)
class ChallengeSummary:
    schema_version: int
    run_id: str
    manifest_path: str
    mode: str
    evaluator_version: str
    baseline_prompt: str
    frontier_prompt: str
    candidate_prompt: str
    baseline_prompt_hash: str
    frontier_prompt_hash: str
    candidate_prompt_hash: str
    primary_pool_fingerprint: str | None
    holdout_pool_fingerprint: str | None
    created_at: str
    primary: ChallengePoolSummary
    holdout: ChallengePoolSummary | None
    promotion_ready: bool
    promotion_reason: str


def run_frontier_challenge(
    *,
    eval_pack_path: str,
    mode: str,
    candidate_prompt_path: str,
    agent_command: str,
    output_root: str | None = None,
    agent_timeout_seconds: int | None = None,
    checks_timeout_seconds: int | None = None,
) -> ChallengeSummary:
    manifest = load_frontier_manifest(eval_pack_path)
    mode_config = resolve_mode(manifest, mode)
    candidate_path = Path(candidate_prompt_path).expanduser().resolve()
    output_base = Path(output_root) if output_root else Path("runs")
    challenge_run_id = build_challenge_id(Path(eval_pack_path).resolve().name, mode)
    challenge_root = output_base / challenge_run_id
    challenge_root.mkdir(parents=True, exist_ok=False)

    baseline_text = Path(mode_config.baseline_prompt).read_text(encoding="utf-8")
    frontier_text = Path(mode_config.frontier_prompt).read_text(encoding="utf-8")
    candidate_text = candidate_path.read_text(encoding="utf-8")
    evaluator_version = mode_config.evaluator_version or EVALUATOR_VERSION
    baseline_hash = mode_config.baseline_prompt_hash or sha256_text(baseline_text)
    frontier_hash = mode_config.frontier_prompt_hash or sha256_text(frontier_text)
    candidate_hash = sha256_text(candidate_text)

    primary_eval = run_prompt_variants(
        repo_ref=manifest.repo_ref,
        eval_pack_path=eval_pack_path,
        mode=mode,
        agent_command=agent_command,
        prompt_variants=[
            ("baseline", baseline_text),
            ("frontier", frontier_text),
            ("candidate", candidate_text),
        ],
        task_names=mode_config.primary_tasks,
        output_root=str(challenge_root / "primary"),
        run_label=f"{Path(eval_pack_path).resolve().name}-{mode}-primary",
        metadata={
            "evaluator_version": evaluator_version,
            "pool_name": "primary",
            "pool_fingerprint": mode_config.primary_pool_fingerprint or "",
            "baseline_prompt_hash": baseline_hash,
            "frontier_prompt_hash": frontier_hash,
            "candidate_prompt_hash": candidate_hash,
        },
        agent_timeout_seconds=agent_timeout_seconds,
        checks_timeout_seconds=checks_timeout_seconds,
    )
    primary_summary = summarize_pool(primary_eval, mode_config.primary_tasks)

    holdout_summary: ChallengePoolSummary | None = None
    promotion_ready = False
    if primary_summary.candidate_beats_frontier and mode_config.holdout_tasks:
        holdout_eval = run_prompt_variants(
            repo_ref=manifest.repo_ref,
            eval_pack_path=eval_pack_path,
            mode=mode,
            agent_command=agent_command,
            prompt_variants=[
                ("baseline", baseline_text),
                ("frontier", frontier_text),
                ("candidate", candidate_text),
            ],
            task_names=mode_config.holdout_tasks,
            output_root=str(challenge_root / "holdout"),
            run_label=f"{Path(eval_pack_path).resolve().name}-{mode}-holdout",
            metadata={
                "evaluator_version": evaluator_version,
                "pool_name": "holdout",
                "pool_fingerprint": mode_config.holdout_pool_fingerprint or "",
                "baseline_prompt_hash": baseline_hash,
                "frontier_prompt_hash": frontier_hash,
                "candidate_prompt_hash": candidate_hash,
            },
            agent_timeout_seconds=agent_timeout_seconds,
            checks_timeout_seconds=checks_timeout_seconds,
        )
        holdout_summary = summarize_pool(holdout_eval, mode_config.holdout_tasks)
        promotion_ready = holdout_summary.candidate_beats_frontier
    else:
        promotion_ready = primary_summary.candidate_beats_frontier

    reason = promotion_reason(primary_summary, holdout_summary)
    summary = ChallengeSummary(
        schema_version=2,
        run_id=challenge_run_id,
        manifest_path=str(Path(eval_pack_path).expanduser().resolve() / "frontier.json"),
        mode=mode,
        evaluator_version=evaluator_version,
        baseline_prompt=str(Path(mode_config.baseline_prompt).resolve()),
        frontier_prompt=str(Path(mode_config.frontier_prompt).resolve()),
        candidate_prompt=str(candidate_path),
        baseline_prompt_hash=baseline_hash,
        frontier_prompt_hash=frontier_hash,
        candidate_prompt_hash=candidate_hash,
        primary_pool_fingerprint=mode_config.primary_pool_fingerprint,
        holdout_pool_fingerprint=mode_config.holdout_pool_fingerprint,
        created_at=datetime.now(UTC).isoformat(),
        primary=primary_summary,
        holdout=holdout_summary,
        promotion_ready=promotion_ready,
        promotion_reason=reason,
    )
    write_challenge_summary(challenge_root / "challenge_summary.json", summary)
    return summary


def render_challenge_summary(summary: ChallengeSummary) -> str:
    lines: list[str] = []
    lines.append(f"Challenge run: {summary.run_id}")
    lines.append(f"Mode: {summary.mode}")
    lines.append(f"Manifest: `{summary.manifest_path}`")
    lines.append(f"Candidate prompt: `{summary.candidate_prompt}`")
    lines.append(f"Evaluator version: {summary.evaluator_version}")
    lines.append(f"Baseline prompt hash: {short_hash(summary.baseline_prompt_hash)}")
    lines.append(f"Frontier prompt hash: {short_hash(summary.frontier_prompt_hash)}")
    lines.append(f"Candidate prompt hash: {short_hash(summary.candidate_prompt_hash)}")
    if summary.primary_pool_fingerprint:
        lines.append(
            f"Primary pool fingerprint: {short_hash(summary.primary_pool_fingerprint)}"
        )
    if summary.holdout_pool_fingerprint:
        lines.append(
            f"Holdout pool fingerprint: {short_hash(summary.holdout_pool_fingerprint)}"
        )
    lines.append("")
    lines.append("Primary pool")
    lines.extend(render_pool(summary.primary))
    if summary.holdout is not None:
        lines.append("")
        lines.append("Holdout pool")
        lines.extend(render_pool(summary.holdout))
    lines.append("")
    lines.append(f"Promotion ready: {'yes' if summary.promotion_ready else 'no'}")
    lines.append(f"Reason: {summary.promotion_reason}")
    return "\n".join(lines)


def load_challenge_summary(path: str) -> ChallengeSummary:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    holdout_payload = payload.get("holdout")
    return ChallengeSummary(
        schema_version=payload["schema_version"],
        run_id=payload["run_id"],
        manifest_path=payload["manifest_path"],
        mode=payload["mode"],
        evaluator_version=payload.get("evaluator_version", EVALUATOR_VERSION),
        baseline_prompt=payload["baseline_prompt"],
        frontier_prompt=payload["frontier_prompt"],
        candidate_prompt=payload["candidate_prompt"],
        baseline_prompt_hash=payload.get("baseline_prompt_hash", ""),
        frontier_prompt_hash=payload.get("frontier_prompt_hash", ""),
        candidate_prompt_hash=payload.get("candidate_prompt_hash", ""),
        primary_pool_fingerprint=payload.get("primary_pool_fingerprint"),
        holdout_pool_fingerprint=payload.get("holdout_pool_fingerprint"),
        created_at=payload["created_at"],
        primary=ChallengePoolSummary(**payload["primary"]),
        holdout=ChallengePoolSummary(**holdout_payload) if holdout_payload else None,
        promotion_ready=payload["promotion_ready"],
        promotion_reason=payload["promotion_reason"],
    )


def summarize_pool(summary: EvalRunSummary, task_ids: list[str]) -> ChallengePoolSummary:
    successes = count_variant_successes(summary)
    return ChallengePoolSummary(
        task_ids=task_ids,
        eval_run_summary=str(resolve_run_summary_path(summary)),
        variant_successes=successes,
        candidate_beats_frontier=successes.get("candidate", 0) > successes.get("frontier", 0),
    )


def count_variant_successes(summary: EvalRunSummary) -> dict[str, int]:
    successes: dict[str, int] = {}
    for task in summary.tasks:
        for variant in task.variants:
            if variant.success:
                successes[variant.name] = successes.get(variant.name, 0) + 1
            else:
                successes.setdefault(variant.name, 0)
    return successes


def promotion_reason(
    primary: ChallengePoolSummary,
    holdout: ChallengePoolSummary | None,
) -> str:
    if not primary.candidate_beats_frontier:
        return "candidate did not beat the current frontier on the primary pool"
    if holdout is None:
        return "candidate beat the current frontier on the primary pool"
    if not holdout.candidate_beats_frontier:
        return "candidate won the primary pool but failed the holdout retest"
    return "candidate beat the current frontier on both the primary and holdout pools"


def resolve_mode(manifest: FrontierManifest, mode: str) -> FrontierModeConfig:
    mode_config = manifest.modes.get(mode)
    if mode_config is None:
        raise ValueError(
            f"Mode is not configured in frontier manifest: {mode}. "
            "Run `promptforge frontier init` first."
        )
    return mode_config


def render_pool(pool: ChallengePoolSummary) -> list[str]:
    lines = [
        f"- Tasks: {', '.join(pool.task_ids)}",
        f"- Eval run: `{pool.eval_run_summary}`",
    ]
    for variant_name in ("baseline", "frontier", "candidate"):
        lines.append(f"- {variant_name} solved: {pool.variant_successes.get(variant_name, 0)}")
    lines.append(
        f"- Candidate beats frontier: {'yes' if pool.candidate_beats_frontier else 'no'}"
    )
    return lines


def resolve_run_summary_path(summary: EvalRunSummary) -> Path:
    if not summary.tasks:
        raise ValueError("Eval summary contains no tasks.")
    first_task_path = Path(summary.tasks[0].task_path).resolve()
    return first_task_path.parents[1] / "run_summary.json"


def build_challenge_id(eval_pack_name: str, mode: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"challenge-{eval_pack_name}-{mode}-{timestamp}"


def write_challenge_summary(path: Path, summary: ChallengeSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
