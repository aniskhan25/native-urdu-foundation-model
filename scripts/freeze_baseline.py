"""Create or verify a content-addressed baseline lockfile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_state(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": git_output(repo_root, "rev-parse", "HEAD"),
        "branch": git_output(repo_root, "branch", "--show-current"),
        "dirty": bool(git_output(repo_root, "status", "--porcelain")),
    }


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def artifact_records(spec: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    records = []
    seen_ids = set()
    for artifact in spec.get("artifacts", []):
        artifact_id = str(artifact["id"])
        if artifact_id in seen_ids:
            raise ValueError(f"Duplicate artifact id: {artifact_id}")
        seen_ids.add(artifact_id)
        path = resolve_path(str(artifact["path"]), repo_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "id": artifact_id,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ValueError("Baseline specification has no artifacts")
    return records


def load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict) or not spec.get("baseline_id"):
        raise ValueError(f"Invalid baseline specification: {path}")
    return spec


def create_lock(spec_path: Path, repo_root: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    repository = repository_state(repo_root)
    if repository["dirty"]:
        raise RuntimeError("Refusing to freeze a dirty repository; commit or stash changes first")
    return {
        "schema_version": 1,
        "baseline_id": spec["baseline_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "specification_path": str(spec_path.resolve()),
        "specification_sha256": sha256_file(spec_path),
        "repository": repository,
        "artifacts": artifact_records(spec, repo_root),
    }


def verify_lock(spec_path: Path, lock_path: Path, repo_root: Path) -> None:
    spec = load_spec(spec_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("baseline_id") != spec["baseline_id"]:
        raise ValueError("Baseline id does not match lockfile")
    if lock.get("specification_sha256") != sha256_file(spec_path):
        raise ValueError("Baseline specification hash does not match lockfile")
    repository = repository_state(repo_root)
    if repository["dirty"]:
        raise RuntimeError("Repository is dirty")
    if repository["commit"] != lock["repository"]["commit"]:
        raise ValueError("Repository commit does not match lockfile")

    expected = {record["id"]: record for record in artifact_records(spec, repo_root)}
    locked = {record["id"]: record for record in lock.get("artifacts", [])}
    if expected != locked:
        raise ValueError("Artifact metadata does not match lockfile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/baseline_v1.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else repo_root / args.spec
    if args.output:
        output_path = args.output if args.output.is_absolute() else repo_root / args.output
        lock = create_lock(spec_path, repo_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote baseline lock: {output_path}")
        return

    verify_path = args.verify if args.verify.is_absolute() else repo_root / args.verify
    verify_lock(spec_path, verify_path, repo_root)
    print(f"Baseline lock verified: {verify_path}")


if __name__ == "__main__":
    main()
