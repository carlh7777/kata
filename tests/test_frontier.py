from __future__ import annotations

from pathlib import Path

from promptforge.challenge import ChallengePoolSummary, promotion_reason
from promptforge.frontier import FrontierManifest, FrontierModeConfig, render_frontier_manifest


def test_render_frontier_manifest_includes_primary_and_holdout_tasks(tmp_path: Path) -> None:
    manifest = FrontierManifest(
        schema_version=1,
        repo_ref="https://github.com/example/repo.git",
        eval_pack=str(tmp_path),
        updated_at="2026-06-28T00:00:00+00:00",
        modes={
            "contributor": FrontierModeConfig(
                baseline_prompt="/tmp/baseline.md",
                frontier_prompt="/tmp/frontier.md",
                primary_tasks=["task-a", "task-b"],
                holdout_tasks=["task-c"],
                evaluator_version="2026-06-29.v1",
                baseline_prompt_hash="a" * 64,
                frontier_prompt_hash="b" * 64,
                primary_pool_fingerprint="c" * 64,
                holdout_pool_fingerprint="d" * 64,
                frontier_updated_at="2026-06-28T01:00:00+00:00",
                frontier_source="run-123",
            )
        },
    )

    rendered = render_frontier_manifest(manifest, "contributor")

    assert "Primary tasks: task-a, task-b" in rendered
    assert "Holdout tasks: task-c" in rendered
    assert "Frontier source: run-123" in rendered
    assert "Evaluator version: 2026-06-29.v1" in rendered
    assert "Baseline prompt hash: aaaaaaaaaaaa" in rendered
    assert "Primary pool fingerprint: cccccccccccc" in rendered


def test_promotion_reason_explains_holdout_failure() -> None:
    primary = ChallengePoolSummary(
        task_ids=["task-a"],
        eval_run_summary="run_summary.json",
        variant_successes={"frontier": 0, "candidate": 1, "baseline": 0},
        candidate_beats_frontier=True,
    )
    holdout = ChallengePoolSummary(
        task_ids=["task-b"],
        eval_run_summary="run_summary.json",
        variant_successes={"frontier": 1, "candidate": 1, "baseline": 0},
        candidate_beats_frontier=False,
    )

    assert (
        promotion_reason(primary, holdout)
        == "candidate won the primary pool but failed the holdout retest"
    )
