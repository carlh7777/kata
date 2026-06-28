from __future__ import annotations

from pathlib import Path

import pytest

from promptforge.cli import main

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "basic_repo"


def test_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert "generate" in capsys.readouterr().out


def test_generate_renders_prompt_for_local_repo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "promptforge.generator.load_registry",
        lambda _url: {
            "fixture/basic_repo": {"emission_share": 0.01, "trusted_label_pipeline": True}
        },
    )

    exit_code = main(["generate", "--repo", str(FIXTURE_REPO), "--mode", "contributor"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "# PromptForge Contributor Prompt: Basic Repo" in out
    assert "Basic Repo is a small fixture project for PromptForge tests." in out
    assert "No explicit validation commands were extracted" in out
    assert "`/docs/` (repo:.github/CODEOWNERS)" in out
    assert "configured SN74 registry" in out or "trusted_label_pipeline" in out


def test_generate_accepts_registry_url_override(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[str] = []

    def fake_load_registry(url: str):
        seen.append(url)
        return {}

    monkeypatch.setattr("promptforge.generator.load_registry", fake_load_registry)
    main(
        [
            "generate",
            "--repo",
            str(FIXTURE_REPO),
            "--mode",
            "contributor",
            "--registry-url",
            "https://example.com/registry.json",
        ]
    )

    assert seen == ["https://example.com/registry.json"]


def test_eval_stub_reports_eval_pack(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["eval", "--repo", "sparkinfer", "--eval-pack", "evals/sparkinfer"])

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "repo=sparkinfer" in out
    assert "eval_pack=evals/sparkinfer" in out
