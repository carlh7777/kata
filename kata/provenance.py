from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from kata.eval_pack import REQUIRED_FILES

EVALUATOR_VERSION = "2026-06-29.v1"


def sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_path(Path(path).expanduser().resolve())


def sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def sha256_directory(root: str | Path, *, include: list[str] | None = None) -> str:
    root_path = Path(root).expanduser().resolve()
    relative_paths = include or [
        path.relative_to(root_path).as_posix()
        for path in sorted(root_path.rglob("*"))
        if path.is_file()
    ]
    hasher = sha256()
    for relative_path in sorted(relative_paths):
        file_path = root_path / relative_path
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def task_fingerprint(task_root: Path) -> str:
    hasher = sha256()
    for filename in REQUIRED_FILES:
        file_path = task_root / filename
        hasher.update(filename.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def pool_fingerprint(task_roots: list[Path]) -> str:
    hasher = sha256()
    for task_root in sorted(task_roots, key=lambda item: item.name):
        hasher.update(task_root.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(task_fingerprint(task_root).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    return value[:length]
