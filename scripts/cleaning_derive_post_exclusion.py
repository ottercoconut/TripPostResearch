#!/usr/bin/env python3
"""按未恢复帖子清单派生排除库；不改来源，也不登记新的冻结版本。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tourism_ugc_study.cleaning.post_exclusion import derive_post_exclusion


def main() -> None:
    """读取显式路径和排除列，把不含昵称或平台用户 ID 的清单打印到标准输出。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, nargs="+", required=True)
    parser.add_argument("--table-column", dest="table_columns", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    try:
        result = derive_post_exclusion(
            args.source,
            args.exclusions,
            args.table_columns,
            args.output,
            args.manifest,
            revision_id=args.revision_id,
            reason=args.reason,
            code_commit=code_commit,
        )
    except (OSError, ValueError, FileExistsError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
