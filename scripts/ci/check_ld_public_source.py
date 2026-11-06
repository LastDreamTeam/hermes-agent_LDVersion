"""Check Git objects (including the index), never the running bot's private home."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import PurePosixPath

SECRET = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|"
    rb"AIza[A-Za-z0-9_-]{30,}|sk-[A-Za-z0-9_-]{20,}|AQ\.[A-Za-z0-9_-]{25,})|"
    rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
)
PRIVATE_NAMES = {"auth.json", "config.yaml", "runtime.env", ".env", "USER.md",
                 "MEMORY.md", ".git-credentials", ".vaultkey", "gateway_state.json"}
PRIVATE_ROOTS = {".hermes", "profiles", "memories", "memory", "notes", "sessions",
                 "a2a_conversations", "logs", "private", "outbox", "game-projects",
                 "主人事务", "舰队", "交付", "回捞", "发送临时"}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def read_blob(ref: str, name: str) -> bytes:
    prefix = "" if ref == ":index" else ref
    return git("show", f"{prefix}:{name}")


def allowed_source(name: str, policy: dict) -> bool:
    path = PurePosixPath(name)
    if path.parts[0] in PRIVATE_ROOTS or path.name in PRIVATE_NAMES:
        return False
    if name in policy["extra_metadata_paths"]:
        return True
    if path.suffix == ".py":
        return len(path.parts) == 1 or path.parts[0] in policy["python_roots"]
    return path.suffix in {".yaml", ".yml"} and path.parts[0] in policy["yaml_roots"]


def check(ref: str) -> dict:
    policy = json.loads(read_blob(ref, "LD_SOURCE_POLICY.json"))
    base = policy["upstream_base"]
    args = ["diff", "--name-status", "--no-renames", "-z"]
    args += ["--cached", base] if ref == ":index" else [base, ref]
    rows = git(*args).split(b"\0")
    failures = []
    count = 0
    for i in range(0, len(rows) - 1, 2):
        status, name = rows[i].decode(), rows[i + 1].decode()
        count += 1
        if status not in {"A", "M"} or not allowed_source(name, policy):
            failures.append({"path": name, "reason": "outside approved code/metadata scope"})
            continue
        if SECRET.search(read_blob(ref, name)):
            failures.append({"path": name, "reason": "credential pattern; value suppressed"})
    return {"ref": ref, "base": base, "changed_files": count,
            "passed": not failures, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=":index", help="Git ref, or :index for staged data")
    result = check(parser.parse_args().ref)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
