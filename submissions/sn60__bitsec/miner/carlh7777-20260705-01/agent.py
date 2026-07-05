from __future__ import annotations

"""SN60 / Bitsec miner — breadth-ranked, function-focused matcher.

The pinned semantic scorer only counts a finding when title/description pin
the exact `.sol` file, contract, function, exploit mechanism, and impact.
This agent improves recall over whole-file-only passes by:

  * ranking the full repo first, then deeply auditing the top contracts;
  * enumerating external/state-changing functions and steering the model at
    the highest-risk entry points inside each file;
  * normalizing every report into ``Contract.function — <bug>`` form with a
    matcher-complete description (file, contract, function, mechanism, impact).

Self-contained (stdlib only). Reads Solidity from ``project_dir`` and calls
the validator inference proxy only (``x-inference-api-key`` header).
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

SOL_EXT = (".sol", ".vy")
SKIP_DIRS = frozenset(
    {
        "test", "tests", "mock", "mocks", "example", "examples", "script",
        "scripts", "node_modules", "vendor", "lib", "out", "artifacts",
        "cache", "broadcast", "interfaces", "interface",
    }
)
RISK_TERMS = (
    "vault", "pool", "router", "bridge", "oracle", "proxy", "upgrade",
    "govern", "treasury", "staking", "market", "lend", "borrow", "collateral",
    "controller", "strategy", "auction", "admin", "owner", "token", "reward",
)
RISK_SIGS = (
    r"\bdelegatecall\b", r"\.call\s*\{", r"\bselfdestruct\b", r"\btx\.origin\b",
    r"\bassembly\b", r"\bupgradeTo\b", r"\binitialize\b", r"\bonlyOwner\b",
    r"\bwithdraw\b", r"\bredeem\b", r"\bliquidat", r"\bflash", r"\bborrow\b",
    r"\brepay\b", r"\btransferFrom\b", r"\bpermit\b", r"\becrecover\b",
    r"\bgetPrice\b", r"\blatestAnswer\b", r"\bunchecked\b", r"\breentran",
)
RE_CONTRACT = re.compile(
    r"\b(?:contract|library)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
RE_FUNCTION = re.compile(
    r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?:public|external|payable)"
)
RE_STATE_FN = re.compile(
    r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)

MAX_BYTES = 180_000
BREADTH_LIMIT = 6
DEPTH_FUNCTIONS = 3
MAX_OUTPUT = 8
WALL_CLOCK = 210.0
HTTP_TIMEOUT = 140
HTTP_RETRIES = 2

AUDITOR_SYSTEM = (
    "You are an elite smart-contract auditor. Report only genuine HIGH or CRITICAL "
    "bugs with a concrete exploit path and material impact (fund loss, privilege "
    "escalation, insolvency, permanent DoS). Ignore style, gas, and hypotheticals."
)


def agent_main(
    project_dir: str | None = None,
    inference_api: str | None = None,
) -> dict:
    findings: list[dict[str, object]] = []
    root = locate_project(project_dir)
    if root is None:
        return {"vulnerabilities": findings}

    deadline = time.monotonic() + WALL_CLOCK
    ranked = rank_sources(root)
    if not ranked:
        return {"vulnerabilities": findings}

    collected: list[dict[str, object]] = []
    for entry in ranked[:BREADTH_LIMIT]:
        if time.monotonic() >= deadline:
            break
        hot_fns = pick_risky_functions(entry["text"], limit=DEPTH_FUNCTIONS)
        chunk = entry["text"][:14_000]
        prompt = make_audit_prompt(entry["rel"], entry["contracts"], chunk, hot_fns)
        try:
            reply = chat(inference_api, prompt)
        except OSError:
            continue
        valid = set(RE_STATE_FN.findall(entry["text"]))
        for item in decode_findings(reply):
            shaped = shape_finding(item, entry, valid)
            if shaped:
                collected.append(shaped)

    findings = merge_findings(collected)[:MAX_OUTPUT]
    return {"vulnerabilities": findings}


def locate_project(project_dir: str | None) -> Path | None:
    paths: list[str] = []
    if project_dir:
        paths.append(project_dir)
    for key in ("PROJECT_DIR", "PROJECT_PATH", "PROJECT_ROOT", "PROJECT_CODE"):
        val = os.environ.get(key)
        if val:
            paths.append(val)
    paths.extend(["/app/project_code", "/app/project", "/project", "/code", "."])
    for raw in paths:
        try:
            candidate = Path(raw).expanduser().resolve()
        except OSError:
            continue
        if candidate.is_dir() and any_sol(candidate):
            return candidate
    return None


def any_sol(root: Path) -> bool:
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SOL_EXT:
                return True
    except OSError:
        return False
    return False


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def risk_score(path: Path, text: str) -> int:
    score = 0
    low = path.as_posix().lower()
    name = path.name.lower()
    for term in RISK_TERMS:
        if term in name:
            score += 7
        elif term in low:
            score += 2
    for sig in RISK_SIGS:
        score += min(len(re.findall(sig, text, flags=re.IGNORECASE)), 5) * 2
    score += min(text.count("function "), 24)
    if "external" in text or "public" in text:
        score += 3
    return score


def rank_sources(root: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOL_EXT:
            continue
        rel_parts = path.relative_to(root).parts[:-1]
        if any(part.lower() in SKIP_DIRS for part in rel_parts):
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
        except OSError:
            continue
        text = read_file(path)
        if "function" not in text:
            continue
        contracts = RE_CONTRACT.findall(text)
        if not contracts:
            continue
        out.append(
            {
                "path": path,
                "rel": path.relative_to(root).as_posix(),
                "text": text,
                "contracts": contracts,
                "score": risk_score(path, text),
            }
        )
    out.sort(key=lambda e: (-int(e["score"]), str(e["rel"])))
    return out


def pick_risky_functions(text: str, limit: int) -> list[str]:
    ext = RE_FUNCTION.findall(text)
    if ext:
        return ext[:limit]
    all_fns = RE_STATE_FN.findall(text)
    priority = [
        fn
        for fn in all_fns
        if any(
            tok in fn.lower()
            for tok in (
                "withdraw", "deposit", "mint", "burn", "borrow", "repay",
                "liquid", "swap", "claim", "execute", "upgrade", "init",
                "transfer", "approve", "set", "update", "admin",
            )
        )
    ]
    merged = priority + [fn for fn in all_fns if fn not in priority]
    return merged[:limit]


def make_audit_prompt(
    rel: str,
    contracts: list[str],
    source: str,
    focus_functions: list[str],
) -> str:
    names = ", ".join(contracts[:8]) or "Unknown"
    focus = ", ".join(focus_functions) if focus_functions else "(scan all functions)"
    return "\n".join(
        [
            f"Audit `{rel}` (contracts: {names}).",
            f"Prioritize these entry points: {focus}.",
            "Find HIGH/CRITICAL exploitable issues only.",
            "Return STRICT JSON:",
            '{"findings": [{"title": "<Contract>.<function> — <bug>", '
            '"contract": "<Name>", "function": "<fn>", '
            f'"file": "{rel}", "severity": "high|critical", '
            '"mechanism": "<precondition -> attack -> outcome>", '
            '"impact": "<concrete harm>", '
            '"description": "<names file, contract, function; mechanism; impact>"}]}',
            "Max 2 findings; empty list if none. Do not invent symbols absent below.",
            "----- SOURCE -----",
            source,
        ]
    )


def chat(inference_api: str | None, user_prompt: str) -> str:
    base = (inference_api or os.environ.get("INFERENCE_API") or "").rstrip("/")
    if not base:
        raise ValueError("missing INFERENCE_API")
    key = os.environ.get("INFERENCE_API_KEY", "").strip()
    payload = json.dumps(
        {
            "messages": [
                {"role": "system", "content": AUDITOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 7500,
        }
    ).encode()
    headers = {"Content-Type": "application/json", "x-inference-api-key": key}
    err: Exception | None = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            req = urllib.request.Request(
                base + "/inference", data=payload, method="POST", headers=headers
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                body = json.loads(resp.read().decode())
            return extract_text(body)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            err = exc
            if attempt < HTTP_RETRIES:
                time.sleep(1 + attempt)
    raise OSError(str(err))


def extract_text(body: dict) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def decode_findings(raw: str) -> list[dict[str, object]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    obj: dict | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            obj = parsed
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            depth = 0
            for idx in range(start, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            maybe = json.loads(text[start : idx + 1])
                            if isinstance(maybe, dict):
                                obj = maybe
                        except json.JSONDecodeError:
                            pass
                        break
    if not obj:
        return []
    items = obj.get("findings") or obj.get("vulnerabilities")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def shape_finding(
    raw: dict[str, object],
    entry: dict[str, object],
    valid_functions: set[str],
) -> dict[str, object] | None:
    sev = str(raw.get("severity") or "").lower()
    if sev not in {"high", "critical"}:
        return None
    contract = str(
        raw.get("contract") or (entry["contracts"][0] if entry["contracts"] else "")
    ).strip()
    function = str(raw.get("function") or "").strip().split(".")[-1].strip("()")
    if function and function not in valid_functions:
        function = ""
    rel = str(raw.get("file") or entry["rel"]).strip() or str(entry["rel"])
    mechanism = str(raw.get("mechanism") or "").strip()
    impact = str(raw.get("impact") or "").strip()
    description = str(raw.get("description") or "").strip()
    title = str(raw.get("title") or "").strip()
    anchor = f"{contract}.{function}" if contract and function else contract or function
    if not title and anchor:
        title = f"{anchor} — {sev} vulnerability"
    elif anchor and anchor.lower() not in title.lower():
        title = f"{anchor} — {title}"
    if len(description) < 90 or (function and function not in description):
        parts = [f"In `{rel}`"]
        if contract:
            parts.append(f"contract `{contract}`")
        if function:
            parts.append(f"function `{function}()`")
        sentence = ", ".join(parts) + "."
        if mechanism:
            sentence += f" Mechanism: {mechanism.rstrip('.')}."
        if impact:
            sentence += f" Impact: {impact.rstrip('.')}."
        if len(sentence) > len(description):
            description = sentence
    if len(description) < 90:
        return None
    return {
        "title": title[:220],
        "description": description,
        "severity": sev,
        "file": rel,
        "function": function,
        "line": raw.get("line") if isinstance(raw.get("line"), int) else None,
        "type": str(raw.get("type") or "logic"),
        "confidence": 0.92 if sev == "critical" else 0.82,
    }


def merge_findings(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    ordered = sorted(
        items,
        key=lambda f: (f.get("severity") == "critical", float(f.get("confidence", 0))),
        reverse=True,
    )
    out: list[dict[str, object]] = []
    for finding in ordered:
        key = (
            str(finding.get("file", "")).lower(),
            str(finding.get("function") or finding.get("title", "")).lower()[:48],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(finding)
    return out


if __name__ == "__main__":
    import sys

    print(json.dumps(agent_main(sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
