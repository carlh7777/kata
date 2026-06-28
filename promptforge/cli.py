from __future__ import annotations

import argparse
from collections.abc import Sequence

from promptforge.generator import generate_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptforge",
        description="Generate and evaluate repo-specific coding-agent prompts.",
    )
    parser.add_argument("--version", action="version", version="promptforge 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a repo-specific prompt.")
    generate.add_argument("--repo", required=True, help="Path or URL of the target repo.")
    generate.add_argument(
        "--mode",
        choices=["contributor", "reviewer"],
        default="contributor",
        help="Prompt mode to generate.",
    )
    generate.add_argument(
        "--registry-url",
        default=None,
        help="Optional SN74 registry JSON URL. Defaults to env or built-in live test-branch URL.",
    )
    generate.set_defaults(handler=handle_generate)

    baseline = subparsers.add_parser("baseline", help="Create or print a baseline prompt.")
    baseline.add_argument("--repo", required=True, help="Path or URL of the target repo.")
    baseline.set_defaults(handler=handle_baseline)

    eval_cmd = subparsers.add_parser("eval", help="Run baseline vs generated prompt evals.")
    eval_cmd.add_argument("--repo", required=True, help="Path or URL of the target repo.")
    eval_cmd.add_argument("--eval-pack", required=True, help="Path to the repo eval pack.")
    eval_cmd.set_defaults(handler=handle_eval)

    report = subparsers.add_parser("report", help="Render an eval report.")
    report.add_argument("--run", required=True, help="Run id or path to run artifacts.")
    report.set_defaults(handler=handle_report)

    return parser


def handle_generate(args: argparse.Namespace) -> int:
    print(generate_prompt(args.repo, args.mode, args.registry_url))
    return 0


def handle_baseline(args: argparse.Namespace) -> int:
    print(
        "Baseline prompt support is not implemented yet.\n"
        f"repo={args.repo}"
    )
    return 2


def handle_eval(args: argparse.Namespace) -> int:
    print(
        "Prompt eval runner is not implemented yet.\n"
        f"repo={args.repo}\n"
        f"eval_pack={args.eval_pack}"
    )
    return 2


def handle_report(args: argparse.Namespace) -> int:
    print(
        "Objective score reports are not implemented yet.\n"
        f"run={args.run}"
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
