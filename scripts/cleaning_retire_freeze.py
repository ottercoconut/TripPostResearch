#!/usr/bin/env python3
"""解除指定研究冻结，并按帖子 ID 从冻结根内删除这些帖。

不读取冻结根以外的研究数据目录。默认只在显式确认后写盘；``--dry-run``
只打印计划和计数。替换库必须是状态为 DERIVED_NOT_FROZEN 的派生结果。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tourism_ugc_study.cleaning.freeze_retirement import retire_freeze

_ROLES = {"raw", "topic", "candidates"}


def _roles(values: list[str]) -> dict[str, Path]:
    """解析重复的 ``role=path``。角色重复或写错时拒绝，避免静默覆盖。"""

    found: dict[str, Path] = {}
    for value in values:
        role, separator, raw = value.partition("=")
        if separator != "=" or role not in _ROLES or not raw or role in found:
            raise ValueError("freeze_retire_replacement_role_invalid")
        found[role] = Path(raw)
    return found


def main() -> None:
    """读取显式路径并退役冻结。成功时把只含路径、计数和哈希的摘要打到标准输出。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--post-ids", required=True, help="JSON 文件，或 {\"ids\":[...]} 文本")
    parser.add_argument("--replacement", action="append", required=True, help="role=path，role 为 raw、topic、candidates")
    parser.add_argument("--replacement-manifest", action="append", required=True, help="role=revision-manifest.json")
    parser.add_argument("--media-db", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-retire", default=None)
    args = parser.parse_args()
    try:
        result = retire_freeze(
            args.workspace,
            args.manifest,
            args.expected_sha256,
            args.post_ids,
            _roles(args.replacement),
            _roles(args.replacement_manifest),
            args.media_db,
            args.backup_root,
            args.reason,
            confirm_retire=args.confirm_retire,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, FileNotFoundError, sqlite3.Error, json.JSONDecodeError, NotImplementedError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
