from __future__ import annotations

from typing import Any

from promptforge.analyzers import (
    dedupe_facts,
    discover_repo_sources,
    extract_commands,
    extract_protected_paths,
    extract_rules,
    extract_summary,
    extract_title,
    format_source_path,
    read_text,
)
from promptforge.config import resolve_registry_url
from promptforge.models import PromptData, SourceFact
from promptforge.registry import load_registry
from promptforge.repository import RepositoryContext, resolve_repository


def generate_prompt(repo_ref: str, mode: str, registry_url: str | None = None) -> str:
    resolved_registry_url = resolve_registry_url(registry_url)
    with resolve_repository(repo_ref) as repo:
        registry = load_registry(resolved_registry_url)
        prompt_data = analyze_repository(repo, registry, resolved_registry_url)
        return render_prompt(prompt_data, mode)


def analyze_repository(
    repo: RepositoryContext,
    registry: dict[str, Any],
    registry_url: str,
) -> PromptData:
    discovered = discover_repo_sources(repo)
    readme_path = discovered["readme"]
    contributing_path = discovered["contributing"]
    agents_path = discovered["agents"]
    codeowners_path = discovered["codeowners"]
    local_weights_path = discovered["local_weights"]
    workflow_paths = list(discovered["workflows"] or [])

    readme_text = read_text(readme_path)
    contributing_text = read_text(contributing_path)
    agents_text = read_text(agents_path)
    codeowners_text = read_text(codeowners_path)

    title = extract_title(readme_text) or repo.display_name
    prompt_data = PromptData(
        title=title,
        repo_display_name=repo.display_name,
        github_full_name=repo.full_name,
    )
    if readme_path is not None:
        prompt_data.summary = extract_summary(readme_text, format_source_path(repo, readme_path))
    if contributing_path is not None:
        prompt_data.rules.extend(
            extract_rules(contributing_text, format_source_path(repo, contributing_path))
        )
        prompt_data.commands.extend(
            extract_commands(contributing_text, format_source_path(repo, contributing_path))
        )
    if agents_path is not None:
        prompt_data.rules.extend(extract_rules(agents_text, format_source_path(repo, agents_path)))
    for workflow_path in workflow_paths:
        prompt_data.commands.extend(
            extract_commands(read_text(workflow_path), format_source_path(repo, workflow_path))
        )
    if codeowners_path is not None:
        prompt_data.protected_paths.extend(
            extract_protected_paths(codeowners_text, format_source_path(repo, codeowners_path))
        )
    prompt_data.rules = dedupe_facts(prompt_data.rules, limit=8)
    prompt_data.commands = dedupe_facts(prompt_data.commands, limit=10)
    prompt_data.protected_paths = dedupe_facts(prompt_data.protected_paths, limit=12)

    registry_entry = registry.get(repo.full_name) if repo.full_name else None
    prompt_data.registry_notes.extend(
        build_registry_notes(repo, registry_entry, local_weights_path is not None, registry_url)
    )
    prompt_data.unknowns.extend(
        collect_unknowns(
            has_contributing=contributing_path is not None,
            has_codeowners=codeowners_path is not None,
            has_workflows=bool(workflow_paths),
            has_registry=registry_entry is not None,
        )
    )
    prompt_data.sources = collect_sources(
        repo=repo,
        readme_path=readme_path,
        contributing_path=contributing_path,
        agents_path=agents_path,
        codeowners_path=codeowners_path,
        local_weights_path=local_weights_path,
        workflow_paths=workflow_paths,
        registry_url=registry_url,
    )
    return prompt_data


def render_prompt(prompt_data: PromptData, mode: str) -> str:
    lines: list[str] = []
    lines.append(f"# PromptForge {mode.capitalize()} Prompt: {prompt_data.title}")
    lines.append("")
    lines.append(f"Repo: `{prompt_data.repo_display_name}`")
    if prompt_data.github_full_name:
        lines.append(f"GitHub: `{prompt_data.github_full_name}`")
    lines.append("")
    lines.append("This prompt is source-grounded from repo files and the configured SN74 registry.")
    lines.append("")
    lines.append("## Repo Overview")
    if prompt_data.summary is not None:
        lines.append(f"- {prompt_data.summary.value} ({prompt_data.summary.source})")
    else:
        lines.append("- No reliable README summary was extracted.")
    lines.append("")
    lines.append("## Contribution Rules")
    if prompt_data.rules:
        lines.extend(f"- {fact.value} ({fact.source})" for fact in prompt_data.rules)
    else:
        lines.append(
            "- No explicit contribution rules were extracted from CONTRIBUTING.md or AGENTS.md."
        )
    lines.append("")
    lines.append("## Validation Commands")
    if prompt_data.commands:
        lines.extend(f"- `{fact.value}` ({fact.source})" for fact in prompt_data.commands)
    else:
        lines.append(
            "- No explicit validation commands were extracted from CONTRIBUTING.md or workflows."
        )
    lines.append("")
    lines.append("## Protected Paths")
    if prompt_data.protected_paths:
        lines.extend(f"- {fact.value} ({fact.source})" for fact in prompt_data.protected_paths)
    else:
        lines.append("- No CODEOWNERS protected paths were extracted.")
    lines.append("")
    lines.append("## Scoring / Registry Notes")
    lines.extend(f"- {fact.value} ({fact.source})" for fact in prompt_data.registry_notes)
    lines.append("")
    lines.append("## Unknowns / Caveats")
    lines.extend(f"- {item}" for item in prompt_data.unknowns)
    lines.append("")
    lines.append("## Sources")
    lines.extend(f"- {source}" for source in prompt_data.sources)
    return "\n".join(lines)


def build_registry_notes(
    repo: RepositoryContext,
    registry_entry: Any,
    has_local_weights: bool,
    registry_url: str,
) -> list[SourceFact]:
    source = registry_url
    notes: list[SourceFact] = []
    if registry_entry is None:
        notes.append(SourceFact("No matching SN74 registry entry was found for this repo.", source))
    else:
        notes.append(SourceFact(f"Registry entry found for `{repo.full_name}`.", source))
        for key in ("emission_share", "fixed_base_score", "trusted_label_pipeline"):
            if key in registry_entry:
                notes.append(SourceFact(f"`{key}`: `{registry_entry[key]}`", source))
        label_multipliers = registry_entry.get("label_multipliers")
        if isinstance(label_multipliers, dict) and label_multipliers:
            preview = ", ".join(
                f"{key}={value}" for key, value in list(label_multipliers.items())[:6]
            )
            notes.append(SourceFact(f"`label_multipliers`: {preview}", source))
        eligibility = registry_entry.get("eligibility")
        if isinstance(eligibility, dict) and eligibility:
            preview = ", ".join(f"{key}={value}" for key, value in eligibility.items())
            notes.append(SourceFact(f"`eligibility`: {preview}", source))
    if has_local_weights:
        notes.append(
            SourceFact(
                "Repo-local `.gittensor/weights.json` also exists and should be reviewed "
                "with the registry.",
                "repo:.gittensor/weights.json",
            )
        )
    return notes


def collect_unknowns(
    *,
    has_contributing: bool,
    has_codeowners: bool,
    has_workflows: bool,
    has_registry: bool,
) -> list[str]:
    unknowns: list[str] = []
    if not has_contributing:
        unknowns.append("No CONTRIBUTING.md was found.")
    if not has_codeowners:
        unknowns.append("No CODEOWNERS file was found.")
    if not has_workflows:
        unknowns.append("No GitHub workflows were found.")
    if not has_registry:
        unknowns.append("No configured SN74 registry entry matched this repo.")
    if not unknowns:
        unknowns.append("No major source gaps were detected in the current scan.")
    return unknowns


def collect_sources(
    *,
    repo: RepositoryContext,
    readme_path,
    contributing_path,
    agents_path,
    codeowners_path,
    local_weights_path,
    workflow_paths: list,
    registry_url: str,
) -> list[str]:
    sources: list[str] = []
    if repo.source_url:
        sources.append(repo.source_url)
    for path in (
        readme_path,
        contributing_path,
        agents_path,
        codeowners_path,
        local_weights_path,
    ):
        if path is not None:
            sources.append(format_source_path(repo, path))
    sources.extend(format_source_path(repo, path) for path in workflow_paths)
    sources.append(registry_url)
    return dedupe_strings(sources)


def dedupe_strings(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
