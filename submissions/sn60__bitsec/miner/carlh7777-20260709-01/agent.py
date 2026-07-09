from __future__ import annotations

"""SN60 miner: repo-map triage + per-function audits + multi-file batch depth.

Spends inference budget on (1) repository target selection, (2) focused audits of
the highest-risk individual functions in the top contracts, and (3) a consolidated
multi-file deep pass. Every finding is normalized to matcher shape:
Contract.function — bug with exact file, mechanism, and impact.

Stdlib only; inference via validator proxy (x-inference-api-key).
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SRC_EXT = (".sol", ".vy")
SKIP_PARTS = frozenset(
    {
        "test", "tests", "mock", "mocks", "example", "examples", "script",
        "scripts", "node_modules", "vendor", "lib", "out", "artifacts",
        "cache", "broadcast", "interfaces", "interface", ".git",
    }
)
FN_SOL = re.compile(
    r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)",
    re.MULTILINE,
)
FN_VY = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
TYPE_SOL = re.compile(
    r"^\s*(?:abstract\s+contract|contract|library|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
IMPORT_SOL = re.compile(r'^\s*import\b[^;]*?["\']([^"\']+)["\']', re.MULTILINE)

MAX_FILE_BYTES = 240_000
MAX_FILES = 60
MAX_FINDINGS = 9
WALL_SEC = 225.0
HTTP_TIMEOUT = 145
MAP_CHARS = 16_000
FN_SNIPPET = 9_500
BATCH_CHARS = 28_000

RISK_IN_NAME = (
    "vault", "pool", "router", "bridge", "oracle", "proxy", "upgrade", "govern",
    "treasury", "staking", "market", "lend", "borrow", "collateral", "controller",
    "strategy", "auction", "token", "reward", "stable", "curve", "liquidity",
)
RISK_IN_CODE = (
    "delegatecall", ".call{", "selfdestruct", "tx.origin", "assembly", "upgradeTo",
    "initialize", "onlyOwner", "withdraw", "redeem", "liquidat", "flash", "borrow",
    "repay", "transferFrom", "permit", "ecrecover", "getPrice", "latestRoundData",
    "slot0", "unchecked", "reentran", "add_liquidity", "remove_liquidity", "get_dy",
    "exchange", "virtual_price", "amplification", "admin_fee", "vesting", "listing",
    "releaseRate", "stepsClaimed",
)

SYSTEM = (
    "You are an expert smart-contract auditor. Report only genuine HIGH or CRITICAL "
    "bugs with a concrete exploit path and material impact. Ignore style, gas, and "
    "speculation. Return strict JSON only; keep answers short."
)


def agent_main(
    project_dir: str | None = None,
    inference_api: str | None = None,
) -> dict:
    findings: list[dict[str, Any]] = []
    started = time.monotonic()
    root = resolve_root(project_dir)
    if root is None:
        return {"vulnerabilities": findings}

    catalog = build_catalog(root)
    if not catalog:
        return {"vulnerabilities": findings}

    rel_index = {item["rel"]: item for item in catalog}
    name_index = {Path(item["rel"]).name: item for item in catalog}
    raw: list[dict[str, Any]] = []

    if time.monotonic() - started < WALL_SEC:
        targets, triage_hits = repo_triage(inference_api, catalog)
        raw.extend(triage_hits)
        ordered = order_targets(targets, catalog)
    else:
        ordered = catalog

    # Per-function micro-audits on the two hottest files.
    for item in ordered[:2]:
        if time.monotonic() - started >= WALL_SEC:
            break
        for fn_name in pick_functions(item, limit=2):
            if time.monotonic() - started >= WALL_SEC:
                break
            raw.extend(audit_single_function(inference_api, item, fn_name, name_index))

    # One multi-file batch on top triage targets.
    if time.monotonic() - started < WALL_SEC:
        raw.extend(audit_batch(inference_api, ordered[:4], name_index))

    for candidate in raw:
        shaped = normalize_finding(candidate, rel_index)
        if shaped is not None:
            findings.append(shaped)

    findings = rank_unique(findings)[:MAX_FINDINGS]
    return {"vulnerabilities": findings}


def resolve_root(project_dir: str | None) -> Path | None:
    options: list[str] = []
    if project_dir:
        options.append(project_dir)
    for key in ("PROJECT_DIR", "PROJECT_PATH", "PROJECT_ROOT", "PROJECT_CODE"):
        val = os.environ.get(key)
        if val:
            options.append(val)
    options.extend(["/app/project_code", "/app/project", "/project", "/code", "."])
    for raw in options:
        try:
            path = Path(raw).expanduser().resolve()
        except OSError:
            continue
        if path.is_dir() and has_contract_sources(path):
            return path
    return None


def has_contract_sources(root: Path) -> bool:
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SRC_EXT:
                return True
    except OSError:
        return False
    return False


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def line_number(text: str, needle: str) -> int | None:
    if not needle:
        return None
    pos = text.find(needle)
    if pos < 0:
        return None
    return text.count("\n", 0, pos) + 1


def list_functions(text: str) -> set[str]:
    names = set(FN_SOL.findall(text))
    names.update(FN_VY.findall(text))
    return names


def score_file(rel: str, text: str) -> int:
    low_path = rel.lower()
    low = text.lower()
    score = min(low.count("function ") + low.count("\ndef "), 30)
    for term in RISK_IN_NAME:
        if term in low_path:
            score += 8
    for sig in RISK_IN_CODE:
        score += min(len(re.findall(re.escape(sig), low, flags=re.IGNORECASE)), 4) * 3
    if "external" in low or "public" in low:
        score += 4
    if "nonreentrant" not in low and any(x in low for x in ("withdraw", "redeem", ".call{")):
        score += 6
    return score


def build_catalog(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SRC_EXT:
            continue
        try:
            rel_path = path.relative_to(root)
            if any(part.lower() in SKIP_PARTS for part in rel_path.parts[:-1]):
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        text = read_text(path)
        if "function" not in text and "\ndef " not in text and "contract " not in text:
            continue
        contracts = TYPE_SOL.findall(text)
        if not contracts and path.suffix.lower() == ".vy":
            contracts = [path.stem]
        if not contracts:
            continue
        rel = rel_path.as_posix()
        out.append(
            {
                "path": path,
                "rel": rel,
                "text": text,
                "contracts": contracts,
                "functions": sorted(list_functions(text)),
                "score": score_file(rel, text),
            }
        )
    out.sort(key=lambda x: (-int(x["score"]), str(x["rel"])))
    return out[:MAX_FILES]


def compact_map(catalog: list[dict[str, Any]]) -> str:
    rows = []
    for item in catalog[:35]:
        rows.append(
            json.dumps(
                {
                    "file": item["rel"],
                    "contracts": item["contracts"][:6],
                    "score": item["score"],
                    "functions": item["functions"][:20],
                },
                separators=(",", ":"),
            )
        )
    return "\n".join(rows)[:MAP_CHARS]


def infer(inference_api: str | None, messages: list[dict[str, str]], max_tokens: int) -> str:
    base = (inference_api or os.environ.get("INFERENCE_API") or "").rstrip("/")
    if not base:
        raise RuntimeError("INFERENCE_API missing")
    payload = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "reasoning": {"effort": "low", "exclude": True},
        }
    ).encode()
    headers = {
        "Content-Type": "application/json",
        "x-inference-api-key": os.environ.get("INFERENCE_API_KEY", ""),
    }
    err: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                base + "/inference", data=payload, method="POST", headers=headers
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8", "replace"))
            return pull_content(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise
            err = exc
        except (OSError, ValueError, TimeoutError) as exc:
            err = exc
        if attempt < 2:
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(str(err))


def pull_content(body: dict[str, Any]) -> str:
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
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    return ""


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = esc = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(stripped[start : i + 1])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def extract_findings(obj: dict[str, Any]) -> list[dict[str, Any]]:
    items = obj.get("findings") or obj.get("vulnerabilities") or []
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def repo_triage(
    inference_api: str | None, catalog: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    prompt = (
        "Study this repository map. Pick files most likely to hold exploitable "
        "high/critical bugs and include any strong findings visible from signatures. "
        "Return strict JSON:\n"
        '{"target_files":["path.sol"],"findings":[{"title":"Contract.function - bug",'
        '"file":"path.sol","contract":"Contract","function":"fn","severity":"high|critical",'
        '"mechanism":"precondition -> attack -> effect","impact":"material harm",'
        '"description":"2-4 precise sentences"}]}\n'
        "Prioritize: stableswap/DEX invariant breaks, LP mint/burn mis-accounting, "
        "decimal/rate scaling, slippage bypass, vesting purchase/transfer math, "
        "marketplace listing balance bugs, oracle staleness, missing access control. "
        "Do not invent symbols.\n\n"
        + compact_map(catalog)
    )
    try:
        obj = parse_json_object(
            infer(
                inference_api,
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                4500,
            )
        )
    except Exception:
        return [], []
    targets = obj.get("target_files")
    files = [str(x) for x in targets if isinstance(x, str)] if isinstance(targets, list) else []
    return files, extract_findings(obj)


def order_targets(targets: list[str], catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rel_map = {c["rel"]: c for c in catalog}
    ordered: list[dict[str, Any]] = []
    for target in targets:
        for rel, item in rel_map.items():
            if target == rel or rel.endswith(target) or target.endswith(rel):
                if item not in ordered:
                    ordered.append(item)
                break
    for item in catalog:
        if item not in ordered:
            ordered.append(item)
    return ordered


def pick_functions(item: dict[str, Any], limit: int) -> list[str]:
    text = str(item["text"])
    ranked = []
    for name in item["functions"]:
        low = name.lower()
        weight = 0
        for token in (
            "withdraw", "deposit", "mint", "burn", "borrow", "repay", "liquid",
            "swap", "exchange", "claim", "transfer", "upgrade", "init", "add_liquidity",
            "remove_liquidity", "calc_", "purchase", "list", "vesting",
        ):
            if token in low:
                weight += 3
        ranked.append((weight, name))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [name for _, name in ranked[:limit] if name]


def related_imports(item: dict[str, Any], by_name: dict[str, dict[str, Any]]) -> str:
    chunks: list[str] = []
    for imp in IMPORT_SOL.findall(str(item["text"])):
        base = imp.rsplit("/", 1)[-1]
        other = by_name.get(base)
        if other and other["rel"] != item["rel"]:
            chunks.append(f"// import {other['rel']}\n{str(other['text'])[:2500]}")
        if len(chunks) >= 2:
            break
    return "\n\n".join(chunks)


def audit_single_function(
    inference_api: str | None,
    item: dict[str, Any],
    function: str,
    by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rel = str(item["rel"])
    snippet = str(item["text"])[:FN_SNIPPET]
    extra = related_imports(item, by_name)
    prompt = (
        f"Deep-audit `{rel}` focusing on `{function}()` and its direct callees. "
        "Find only high/critical exploitable issues. Return strict JSON:\n"
        '{"findings":[{"title":"Contract.function - bug","file":"'
        + rel
        + '","contract":"Name","function":"'
        + function
        + '","line":1,"severity":"high|critical","mechanism":"...","impact":"...",'
        '"description":"names file, contract, function, mechanism, impact"}]}\n'
        "Max 2 findings; empty list if none.\n----- SOURCE -----\n"
        + snippet
    )
    if extra:
        prompt += "\n----- IMPORTS -----\n" + extra
    try:
        obj = parse_json_object(
            infer(
                inference_api,
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                5500,
            )
        )
    except Exception:
        return []
    return extract_findings(obj)


def audit_batch(
    inference_api: str | None,
    batch: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not batch:
        return []
    header = (
        "Audit the sources below together. Return strict JSON findings array only. "
        "Each finding must name exact file, contract, function, mechanism, impact.\n"
    )
    parts = [header]
    budget = BATCH_CHARS - len(header)
    for item in batch:
        block = f"\n===== {item['rel']} =====\n{str(item['text'])}\n"
        imp = related_imports(item, by_name)
        if imp:
            block += f"\n===== CONTEXT {item['rel']} =====\n{imp}\n"
        if len(block) > budget:
            block = block[: max(0, budget)] + "\n/* truncated */\n"
        if budget <= 0:
            break
        parts.append(block)
        budget -= len(block)
    try:
        obj = parse_json_object(
            infer(
                inference_api,
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "".join(parts)}],
                7500,
            )
        )
    except Exception:
        return []
    return extract_findings(obj)


def normalize_finding(
    raw: dict[str, Any], rel_index: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    file_val = str(raw.get("file") or raw.get("path") or "").strip()
    if not file_val:
        return None
    record = None
    for rel, item in rel_index.items():
        if file_val == rel or rel.endswith(file_val) or file_val.endswith(rel):
            record = item
            file_val = rel
            break
    if record is None:
        return None

    severity = str(raw.get("severity") or "").lower().strip()
    if severity not in {"high", "critical"}:
        return None

    function = str(raw.get("function") or "").strip().strip("`() ")
    if "." in function:
        function = function.split(".")[-1]
    valid = set(record["functions"])
    if function and function not in valid:
        function = ""

    contract = str(raw.get("contract") or "").strip()
    if not contract and record["contracts"]:
        contract = str(record["contracts"][0])

    mechanism = str(raw.get("mechanism") or "").strip()
    impact = str(raw.get("impact") or "").strip()
    description = str(raw.get("description") or "").strip()
    title = str(raw.get("title") or "").strip()

    if len(mechanism) < 20 and len(description) < 100:
        return None

    anchor = ".".join(x for x in (contract, function) if x)
    if not title:
        title = f"{anchor or file_val} - high severity issue"
    elif anchor and anchor.lower() not in title.lower():
        title = f"{anchor} - {title}"

    where = f"In `{file_val}`"
    if contract:
        where += f", contract `{contract}`"
    if function:
        where += f", function `{function}()`"
    rebuilt = where + ". "
    if mechanism:
        rebuilt += "Mechanism: " + mechanism.rstrip(".") + ". "
    if impact:
        rebuilt += "Impact: " + impact.rstrip(".") + ". "
    if description:
        rebuilt += description
    description = " ".join(rebuilt.split())
    if len(description) < 95:
        return None

    line = raw.get("line")
    if not isinstance(line, int):
        needle = f"function {function}" if function else f"def {function}"
        line = line_number(str(record["text"]), needle)

    return {
        "title": title[:220],
        "description": description[:2800],
        "severity": severity,
        "file": file_val,
        "function": function,
        "line": line if isinstance(line, int) else None,
        "type": "logic",
        "confidence": 0.9 if severity == "critical" else 0.83,
    }


def rank_unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    ordered = sorted(
        items,
        key=lambda f: (
            f.get("severity") == "critical",
            float(f.get("confidence") or 0),
            len(str(f.get("description") or "")),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for f in ordered:
        key = (
            str(f.get("file") or "").lower(),
            str(f.get("function") or "").lower(),
            str(f.get("title") or "").lower()[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


if __name__ == "__main__":
    import sys

    print(json.dumps(agent_main(sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
