from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from kata.evaluators.sn60_bitsec import (
    DEFAULT_REPLICAS_PER_PROJECT,
    Sn60DuelSummary,
    Sn60EvaluationHook,
    Sn60ExecutionHook,
    Sn60SandboxSource,
    Sn60VariantSummary,
    bitsec_project_image,
    hash_bundle_root,
    resolve_sn60_sandbox_source,
    run_sn60_bitsec_duel,
)
from kata.lane_state import (
    BENCHMARK_SNAPSHOT_SCHEMA_VERSION,
    CHALLENGE_STATE_SCHEMA_VERSION,
    PROMOTION_RECORD_SCHEMA_VERSION,
    BenchmarkSnapshotState,
    ChallengeState,
    PromotionRecord,
    write_benchmark_snapshot,
    write_challenge_state,
    write_promotion_record,
)
from kata.live_progress import update_live_status
from kata.provenance import short_hash
from kata.screening import (
    Sn60ScreeningHook,
    Sn60ScreeningResult,
    run_sn60_screening,
    screening_result_payload,
    sn60_screening_freshness_fingerprint,
    write_screening_result,
)

SN60_MINER_LANE_ID = "sn60__bitsec"
SN60_MINER_MODE = "miner"
SN60_VALIDATOR_MODEL = "sn60-bitsec-sandbox"


@dataclass(frozen=True)
class ChallengePoolSummary:
    task_ids: list[str]
    eval_run_summary: str
    total_task_weight: float
    variant_successes: dict[str, int]
    variant_invalid_tasks: dict[str, int]
    variant_scores: dict[str, float]
    candidate_beats_frontier: bool
    candidate_score_delta: float


@dataclass(frozen=True)
class ChallengeSummary:
    schema_version: int
    run_id: str
    manifest_path: str
    mode: str
    evaluator_version: str
    validator_model: str
    frontier_artifact: str
    candidate_artifact: str
    frontier_artifact_hash: str
    candidate_artifact_hash: str
    primary_pool_fingerprint: str | None
    holdout_pool_fingerprint: str | None
    promotion_margin_points: float
    holdout_promotion_margin_points: float
    created_at: str
    primary: ChallengePoolSummary
    holdout: ChallengePoolSummary | None
    promotion_ready: bool
    promotion_reason: str


@dataclass(frozen=True)
class Sn60PromotionDecision:
    promotion_ready: bool
    final_winner: str
    reason: str




def run_sn60_challenge(
    *,
    frontier_artifact_path: str,
    candidate_artifact_path: str,
    project_keys: list[str],
    candidate_submission_id: str,
    lane_id: str = SN60_MINER_LANE_ID,
    output_root: str | None = None,
    replicas_per_project: int = DEFAULT_REPLICAS_PER_PROJECT,
    sandbox_root: str | None = None,
    benchmark_file: str | None = None,
    sandbox_commit: str | None = None,
    screening_result: dict[str, object] | None = None,
    public_root: str | None = None,
    screening_hook: Sn60ScreeningHook | None = None,
    execution_hook: Sn60ExecutionHook | None = None,
    evaluation_hook: Sn60EvaluationHook | None = None,
) -> ChallengeSummary:
    if not project_keys:
        raise ValueError("SN60 challenge requires at least one screening project key.")
    sandbox_source = resolve_sn60_sandbox_source(
        sandbox_root=sandbox_root,
        benchmark_file=benchmark_file,
        sandbox_commit=sandbox_commit,
        scorer_version="ScaBenchScorerV2",
    )
    update_live_status(
        {
            "state": "screening",
            "phase": "sn60-screening",
            "lane_id": lane_id,
            "candidate_submission_id": candidate_submission_id,
            "project_keys": list(project_keys),
        }
    )
    screening = run_sn60_screening(
        candidate_artifact_path=candidate_artifact_path,
        project_key=project_keys[0],
        output_root=output_root or "runs",
        sandbox_source=sandbox_source,
        execution_hook=screening_hook,
    )
    effective_screening_result = screening_result_payload(screening)
    if screening_result:
        effective_screening_result["details"] = {
            **dict(effective_screening_result.get("details") or {}),
            "caller_context": screening_result,
        }
    if not screening.passed:
        update_live_status(
            {
                "state": "verifying",
                "phase": "verifying",
                "lane_id": lane_id,
                "promotion_ready": False,
                "promotion_reason": "candidate failed SN60 screening",
            }
        )
        summary = build_sn60_screening_failure_summary(
            frontier_artifact_path=frontier_artifact_path,
            candidate_artifact_path=candidate_artifact_path,
            project_keys=project_keys,
            lane_id=lane_id,
            screening=screening,
        )
        write_challenge_summary(
            Path(screening.result_path).with_name("challenge_summary.json"),
            summary,
        )
        record_sn60_screening_failure_provenance(
            lane_id=lane_id,
            candidate_submission_id=candidate_submission_id,
            frontier_artifact_path=frontier_artifact_path,
            project_keys=project_keys,
            replicas_per_project=replicas_per_project,
            screening=screening,
            public_root=public_root,
        )
        return summary

    update_live_status(
        {
            "state": "evaluating",
            "phase": "sn60-duel",
            "lane_id": lane_id,
            "candidate_submission_id": candidate_submission_id,
            "project_keys": list(project_keys),
            "replicas_per_project": replicas_per_project,
        }
    )
    duel_summary = run_sn60_bitsec_duel(
        frontier_artifact_path=frontier_artifact_path,
        candidate_artifact_path=candidate_artifact_path,
        project_keys=project_keys,
        output_root=output_root,
        replicas_per_project=replicas_per_project,
        sandbox_root=sandbox_source.sandbox_root,
        benchmark_file=sandbox_source.benchmark_file,
        sandbox_commit=sandbox_source.sandbox_commit,
        execution_hook=execution_hook,
        evaluation_hook=evaluation_hook,
    )
    write_screening_result(Path(duel_summary.output_root) / "screening_result.json", screening)
    summary = sn60_duel_to_challenge_summary(
        duel_summary,
        lane_id=lane_id,
        screening_result=effective_screening_result,
    )
    challenge_summary_path = Path(duel_summary.output_root) / "challenge_summary.json"
    write_challenge_summary(challenge_summary_path, summary)
    record_sn60_lane_provenance(
        lane_id=lane_id,
        candidate_submission_id=candidate_submission_id,
        duel_summary=duel_summary,
        screening_result=effective_screening_result,
        public_root=public_root,
    )
    update_live_status(
        {
            "state": "verifying",
            "phase": "verifying",
            "lane_id": lane_id,
            "challenge_summary_path": str(challenge_summary_path),
            "promotion_ready": summary.promotion_ready,
            "promotion_reason": summary.promotion_reason,
        }
    )
    return summary


def sn60_duel_to_challenge_summary(
    duel_summary: Sn60DuelSummary,
    *,
    lane_id: str = SN60_MINER_LANE_ID,
    screening_result: dict[str, object] | None = None,
) -> ChallengeSummary:
    decision = evaluate_sn60_promotion(
        frontier=duel_summary.frontier,
        candidate=duel_summary.candidate,
        screening_result=screening_result,
    )
    freshness_fingerprint = sn60_freshness_fingerprint(duel_summary)
    duel_summary_path = Path(duel_summary.output_root) / "duel_summary.json"
    return ChallengeSummary(
        schema_version=4,
        run_id=duel_summary.run_id,
        manifest_path=str(duel_summary_path),
        mode=SN60_MINER_MODE,
        evaluator_version=sn60_evaluator_version(duel_summary),
        validator_model=SN60_VALIDATOR_MODEL,
        frontier_artifact=duel_summary.frontier.artifact_path,
        candidate_artifact=duel_summary.candidate.artifact_path,
        frontier_artifact_hash=duel_summary.frontier.artifact_hash,
        candidate_artifact_hash=duel_summary.candidate.artifact_hash,
        primary_pool_fingerprint=freshness_fingerprint,
        holdout_pool_fingerprint=None,
        promotion_margin_points=0.0,
        holdout_promotion_margin_points=0.0,
        created_at=duel_summary.created_at,
        primary=sn60_duel_to_pool_summary(
            duel_summary,
            eval_run_summary=duel_summary_path,
            screening_result=screening_result,
        ),
        holdout=None,
        promotion_ready=decision.promotion_ready,
        promotion_reason=f"{lane_id}: {decision.reason}",
    )


def sn60_duel_to_pool_summary(
    duel_summary: Sn60DuelSummary,
    *,
    eval_run_summary: Path,
    screening_result: dict[str, object] | None = None,
) -> ChallengePoolSummary:
    frontier_score = round(duel_summary.frontier.aggregated_score * 100, 2)
    candidate_score = round(duel_summary.candidate.aggregated_score * 100, 2)
    decision = evaluate_sn60_promotion(
        frontier=duel_summary.frontier,
        candidate=duel_summary.candidate,
        screening_result=screening_result,
    )
    return ChallengePoolSummary(
        task_ids=list(duel_summary.project_keys),
        eval_run_summary=str(eval_run_summary),
        total_task_weight=float(len(duel_summary.project_keys)),
        variant_successes={
            "frontier": duel_summary.frontier.codebase_pass_count,
            "candidate": duel_summary.candidate.codebase_pass_count,
        },
        variant_invalid_tasks={
            "frontier": duel_summary.frontier.invalid_runs,
            "candidate": duel_summary.candidate.invalid_runs,
        },
        variant_scores={
            "frontier": frontier_score,
            "candidate": candidate_score,
        },
        candidate_beats_frontier=decision.final_winner == "candidate",
        candidate_score_delta=round(candidate_score - frontier_score, 2),
    )


def build_sn60_screening_failure_summary(
    *,
    frontier_artifact_path: str,
    candidate_artifact_path: str,
    project_keys: list[str],
    lane_id: str,
    screening: Sn60ScreeningResult,
) -> ChallengeSummary:
    frontier_root = Path(frontier_artifact_path).expanduser().resolve()
    candidate_root = Path(candidate_artifact_path).expanduser().resolve()
    frontier_hash = hash_bundle_root(frontier_root)
    freshness_fingerprint = sn60_screening_freshness_fingerprint(
        frontier_artifact_hash=frontier_hash,
        screening_result=screening,
    )
    reason = "; ".join(screening.reasons) if screening.reasons else "unknown screening failure"
    return ChallengeSummary(
        schema_version=4,
        run_id=screening.run_id,
        manifest_path=screening.result_path,
        mode=SN60_MINER_MODE,
        evaluator_version=(
            f"{screening.sandbox_source.scorer_version}"
            f"@{short_hash(screening.sandbox_source.sandbox_commit)}"
        ),
        validator_model=SN60_VALIDATOR_MODEL,
        frontier_artifact=str(frontier_root),
        candidate_artifact=str(candidate_root),
        frontier_artifact_hash=frontier_hash,
        candidate_artifact_hash=screening.artifact_hash,
        primary_pool_fingerprint=freshness_fingerprint,
        holdout_pool_fingerprint=None,
        promotion_margin_points=0.0,
        holdout_promotion_margin_points=0.0,
        created_at=screening.created_at,
        primary=ChallengePoolSummary(
            task_ids=list(project_keys),
            eval_run_summary=screening.result_path,
            total_task_weight=1.0,
            variant_successes={"frontier": 0, "candidate": 0},
            variant_invalid_tasks={"frontier": 0, "candidate": 1},
            variant_scores={"frontier": 0.0, "candidate": 0.0},
            candidate_beats_frontier=False,
            candidate_score_delta=0.0,
        ),
        holdout=None,
        promotion_ready=False,
        promotion_reason=f"{lane_id}: candidate failed SN60 screening: {reason}",
    )


def evaluate_sn60_promotion(
    *,
    frontier: Sn60VariantSummary,
    candidate: Sn60VariantSummary,
    screening_result: dict[str, object] | None = None,
) -> Sn60PromotionDecision:
    screening_status = screening_result.get("status") if screening_result is not None else None
    if screening_result is not None and screening_status not in {"passed", "pass", True}:
        return Sn60PromotionDecision(
            promotion_ready=False,
            final_winner="frontier",
            reason="candidate failed SN60 screening",
        )
    if candidate.invalid_runs > 0:
        return Sn60PromotionDecision(
            promotion_ready=False,
            final_winner="frontier",
            reason="candidate has invalid SN60 replica runs",
        )

    candidate_rank = sn60_variant_rank(candidate)
    frontier_rank = sn60_variant_rank(frontier)
    if candidate_rank <= frontier_rank:
        return Sn60PromotionDecision(
            promotion_ready=False,
            final_winner="frontier",
            reason="candidate did not beat the current SN60 king",
        )
    return Sn60PromotionDecision(
        promotion_ready=True,
        final_winner="candidate",
        reason="candidate beat the current SN60 king",
    )


def sn60_variant_rank(summary: Sn60VariantSummary) -> tuple[float, int, int, int]:
    # Promotion comparator per the frozen SN60 spec:
    # aggregated score first, codebase pass count second, true positives third.
    return (
        round(summary.aggregated_score, 8),
        summary.codebase_pass_count,
        summary.true_positives,
        -summary.invalid_runs,
    )


def record_sn60_lane_provenance(
    *,
    lane_id: str,
    candidate_submission_id: str,
    duel_summary: Sn60DuelSummary,
    screening_result: dict[str, object],
    public_root: str | None = None,
    reward_label_applied: str | None = None,
) -> tuple[Path, Path]:
    decision = evaluate_sn60_promotion(
        frontier=duel_summary.frontier,
        candidate=duel_summary.candidate,
        screening_result=screening_result,
    )
    freshness_fingerprint = sn60_freshness_fingerprint(duel_summary)
    record_sn60_benchmark_snapshot(
        lane_id=lane_id,
        sandbox_source=duel_summary.sandbox_source,
        project_keys=list(duel_summary.project_keys),
        public_root=public_root,
    )
    challenge_path = write_challenge_state(
        lane_id,
        ChallengeState(
            schema_version=CHALLENGE_STATE_SCHEMA_VERSION,
            candidate_submission_id=candidate_submission_id,
            candidate_artifact_hash=duel_summary.candidate.artifact_hash,
            king_artifact_hash=duel_summary.frontier.artifact_hash,
            screening_result=screening_result,
            selected_project_keys=list(duel_summary.project_keys),
            validator_replica_count=duel_summary.replicas_per_project,
            run_ids=[duel_summary.run_id],
            freshness_fingerprint=freshness_fingerprint,
            updated_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )
    promotion_path = write_promotion_record(
        lane_id,
        PromotionRecord(
            schema_version=PROMOTION_RECORD_SCHEMA_VERSION,
            final_metrics=sn60_final_metrics(duel_summary, decision),
            local_replica_scores=sn60_local_replica_scores(duel_summary),
            pass_counts={
                "frontier": duel_summary.frontier.codebase_pass_count,
                "candidate": duel_summary.candidate.codebase_pass_count,
            },
            true_positives={
                "frontier": duel_summary.frontier.true_positives,
                "candidate": duel_summary.candidate.true_positives,
            },
            invalid_runs={
                "frontier": duel_summary.frontier.invalid_runs,
                "candidate": duel_summary.candidate.invalid_runs,
            },
            final_winner=decision.final_winner,
            reward_label_applied=reward_label_applied,
            recorded_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )
    return challenge_path, promotion_path


def record_sn60_screening_failure_provenance(
    *,
    lane_id: str,
    candidate_submission_id: str,
    frontier_artifact_path: str,
    project_keys: list[str],
    replicas_per_project: int,
    screening: Sn60ScreeningResult,
    public_root: str | None = None,
) -> tuple[Path, Path]:
    frontier_hash = hash_bundle_root(Path(frontier_artifact_path).expanduser().resolve())
    freshness_fingerprint = sn60_screening_freshness_fingerprint(
        frontier_artifact_hash=frontier_hash,
        screening_result=screening,
    )
    screening_payload = screening_result_payload(screening)
    reason = "; ".join(screening.reasons) if screening.reasons else "unknown screening failure"
    record_sn60_benchmark_snapshot(
        lane_id=lane_id,
        sandbox_source=screening.sandbox_source,
        project_keys=list(project_keys),
        public_root=public_root,
    )
    challenge_path = write_challenge_state(
        lane_id,
        ChallengeState(
            schema_version=CHALLENGE_STATE_SCHEMA_VERSION,
            candidate_submission_id=candidate_submission_id,
            candidate_artifact_hash=screening.artifact_hash,
            king_artifact_hash=frontier_hash,
            screening_result=screening_payload,
            selected_project_keys=list(project_keys),
            validator_replica_count=replicas_per_project,
            run_ids=[screening.run_id],
            freshness_fingerprint=freshness_fingerprint,
            updated_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )
    promotion_path = write_promotion_record(
        lane_id,
        PromotionRecord(
            schema_version=PROMOTION_RECORD_SCHEMA_VERSION,
            final_metrics={
                "run_id": screening.run_id,
                "promotion_ready": False,
                "promotion_reason": f"candidate failed SN60 screening: {reason}",
                "screening_status": screening.status,
                "screening_stage": screening.stage,
                "sandbox_commit": screening.sandbox_source.sandbox_commit,
                "benchmark_sha256": screening.sandbox_source.benchmark_sha256,
                "scorer_version": screening.sandbox_source.scorer_version,
            },
            local_replica_scores={"frontier": [], "candidate": []},
            pass_counts={"frontier": 0, "candidate": 0},
            true_positives={"frontier": 0, "candidate": 0},
            invalid_runs={"frontier": 0, "candidate": 1},
            final_winner="frontier",
            reward_label_applied=None,
            recorded_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )
    return challenge_path, promotion_path


def sn60_final_metrics(
    duel_summary: Sn60DuelSummary,
    decision: Sn60PromotionDecision,
) -> dict[str, object]:
    frontier_aggregated = duel_summary.frontier.aggregated_score
    candidate_aggregated = duel_summary.candidate.aggregated_score
    return {
        "run_id": duel_summary.run_id,
        "promotion_ready": decision.promotion_ready,
        "promotion_reason": decision.reason,
        "frontier_aggregated_score": frontier_aggregated,
        "candidate_aggregated_score": candidate_aggregated,
        "candidate_aggregated_score_delta": candidate_aggregated - frontier_aggregated,
        # `average score` is defined as exactly equal to `aggregated score` per the
        # SN60 metric naming rule; kept for existing dashboard consumers.
        "frontier_average_score": frontier_aggregated,
        "candidate_average_score": candidate_aggregated,
        "candidate_score_delta": candidate_aggregated - frontier_aggregated,
        "sandbox_commit": duel_summary.sandbox_source.sandbox_commit,
        "benchmark_sha256": duel_summary.sandbox_source.benchmark_sha256,
        "scorer_version": duel_summary.sandbox_source.scorer_version,
    }


def sn60_local_replica_scores(duel_summary: Sn60DuelSummary) -> dict[str, list[float]]:
    return {
        "frontier": [result.score for result in duel_summary.frontier.replica_results],
        "candidate": [result.score for result in duel_summary.candidate.replica_results],
    }


def record_sn60_benchmark_snapshot(
    *,
    lane_id: str,
    sandbox_source: Sn60SandboxSource,
    project_keys: list[str],
    public_root: str | None = None,
) -> None:
    write_benchmark_snapshot(
        lane_id,
        BenchmarkSnapshotState(
            schema_version=BENCHMARK_SNAPSHOT_SCHEMA_VERSION,
            sandbox_mirror_source=sandbox_source.sandbox_root,
            sandbox_commit_hash=sandbox_source.sandbox_commit,
            benchmark_dataset_id=Path(sandbox_source.benchmark_file).name,
            benchmark_dataset_hash=sandbox_source.benchmark_sha256,
            project_list_hash=sn60_project_list_hash(project_keys),
            project_keys=list(project_keys),
            container_images=[
                bitsec_project_image(project_key) for project_key in project_keys
            ],
            scorer_version=sandbox_source.scorer_version,
            updated_at=datetime.now(UTC).isoformat(),
        ),
        public_root=public_root,
    )


def sn60_project_list_hash(project_keys: list[str]) -> str:
    payload = json.dumps(sorted(project_keys))
    return sha256(payload.encode("utf-8")).hexdigest()


def sn60_freshness_fingerprint(duel_summary: Sn60DuelSummary) -> str:
    payload = {
        "frontier_artifact_hash": duel_summary.frontier.artifact_hash,
        "candidate_artifact_hash": duel_summary.candidate.artifact_hash,
        "project_keys": duel_summary.project_keys,
        "replicas_per_project": duel_summary.replicas_per_project,
        "sandbox_commit": duel_summary.sandbox_source.sandbox_commit,
        "benchmark_sha256": duel_summary.sandbox_source.benchmark_sha256,
        "scorer_version": duel_summary.sandbox_source.scorer_version,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def sn60_evaluator_version(duel_summary: Sn60DuelSummary) -> str:
    return (
        f"{duel_summary.sandbox_source.scorer_version}"
        f"@{short_hash(duel_summary.sandbox_source.sandbox_commit)}"
    )




def render_challenge_summary(summary: ChallengeSummary) -> str:
    lines: list[str] = []
    lines.append(f"Challenge run: {summary.run_id}")
    lines.append(f"Mode: {summary.mode}")
    lines.append(f"Manifest: `{summary.manifest_path}`")
    lines.append(f"Candidate artifact: `{summary.candidate_artifact}`")
    lines.append(f"Evaluator version: {summary.evaluator_version}")
    lines.append(f"Validator model: {summary.validator_model}")
    lines.append(f"Frontier artifact hash: {short_hash(summary.frontier_artifact_hash)}")
    lines.append(f"Candidate artifact hash: {short_hash(summary.candidate_artifact_hash)}")
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
    lines.append(f"Promotion margin: {summary.promotion_margin_points:.1f} points")
    lines.append(
        f"Holdout margin: {summary.holdout_promotion_margin_points:.1f} points"
    )
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
        evaluator_version=payload.get("evaluator_version", ""),
        validator_model=payload.get("validator_model", SN60_VALIDATOR_MODEL),
        frontier_artifact=payload["frontier_artifact"],
        candidate_artifact=payload["candidate_artifact"],
        frontier_artifact_hash=payload.get("frontier_artifact_hash", ""),
        candidate_artifact_hash=payload.get("candidate_artifact_hash", ""),
        primary_pool_fingerprint=payload.get("primary_pool_fingerprint"),
        holdout_pool_fingerprint=payload.get("holdout_pool_fingerprint"),
        promotion_margin_points=payload.get("promotion_margin_points", 0.0),
        holdout_promotion_margin_points=payload.get(
            "holdout_promotion_margin_points", 0.0
        ),
        created_at=payload["created_at"],
        primary=parse_challenge_pool(payload["primary"]),
        holdout=parse_challenge_pool(holdout_payload) if holdout_payload else None,
        promotion_ready=payload["promotion_ready"],
        promotion_reason=payload["promotion_reason"],
    )


def parse_challenge_pool(payload: dict[str, object]) -> ChallengePoolSummary:
    variant_scores = payload.get("variant_scores") or {}
    candidate_score = float(variant_scores.get("candidate", 0.0)) if variant_scores else 0.0
    frontier_score = float(variant_scores.get("frontier", 0.0)) if variant_scores else 0.0
    return ChallengePoolSummary(
        task_ids=list(payload["task_ids"]),
        eval_run_summary=str(payload["eval_run_summary"]),
        total_task_weight=float(payload.get("total_task_weight", len(payload["task_ids"]))),
        variant_successes=dict(payload.get("variant_successes") or {}),
        variant_invalid_tasks=dict(payload.get("variant_invalid_tasks") or {}),
        variant_scores={name: float(score) for name, score in variant_scores.items()},
        candidate_beats_frontier=bool(
            payload.get("candidate_beats_frontier", candidate_score > frontier_score)
        ),
        candidate_score_delta=float(
            payload.get("candidate_score_delta", round(candidate_score - frontier_score, 2))
        ),
    )
























def render_pool(pool: ChallengePoolSummary) -> list[str]:
    lines = [
        f"- Tasks: {', '.join(pool.task_ids)}",
        f"- Eval run: `{pool.eval_run_summary}`",
        f"- Total task weight: {pool.total_task_weight:g}",
    ]
    for variant_name in ("frontier", "candidate"):
        lines.append(f"- {variant_name} solved: {pool.variant_successes.get(variant_name, 0)}")
        lines.append(
            f"- {variant_name} invalid tasks: {pool.variant_invalid_tasks.get(variant_name, 0)}"
        )
        lines.append(f"- {variant_name} score: {pool.variant_scores.get(variant_name, 0.0):.2f}")
    lines.append(
        f"- Candidate beats frontier: {'yes' if pool.candidate_beats_frontier else 'no'}"
    )
    lines.append(f"- Candidate score delta: {pool.candidate_score_delta:+.2f}")
    return lines






def write_challenge_summary(path: Path, summary: ChallengeSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
