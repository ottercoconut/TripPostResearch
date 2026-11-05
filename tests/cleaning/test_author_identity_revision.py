"""作者身份派生的合成库测试。不读取 data/、results/ 或真实采集库。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tourism_ugc_study.cleaning.author_identity_revision import derive_author_identity_revision

SECRET_NICK = "LEAKED_NICK_TOKEN"
SECRET_UID = "LEAKED_UID_TOKEN"
PYTHON = Path("/Users/kawauso/Documents/Projects/TripPostResearch/.venv/bin/python")


def _digest(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:16]


def _mask(value: str) -> str:
    if len(value) <= 1:
        return "*"
    if len(value) == 2:
        return value[0] + "*"
    return value[0] + "***" + value[-1]


def _author(creator: str, nickname: str, **extra: object) -> str:
    payload = {"creator_hash": creator, "nickname": nickname, "followers_count": 12, "profile": {"city": "青岛"}}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _raw(creator: str, *, platform: str, nickname: str, user_nickname: str | None = None, source: str) -> str:
    payload: dict[str, object] = {
        "content_detail_source": source,
        "creator_hash": creator,
        "extra_flag": True,
    }
    if platform == "zhihu":
        payload["nickname"] = "残留昵称键"
        payload["user_nickname"] = user_nickname
    else:
        payload["nickname"] = nickname
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _create_database(
    path: Path,
    rows: list[tuple[object, ...]],
    *,
    images: list[tuple[object, ...]] | None = None,
    waivers: list[tuple[object, ...]] | None = None,
) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE source_platforms(platform_key TEXT PRIMARY KEY, label TEXT NOT NULL);
            CREATE TABLE web_posts(
                id INTEGER PRIMARY KEY,
                platform_key TEXT NOT NULL REFERENCES source_platforms(platform_key),
                platform_post_id TEXT NOT NULL,
                author_platform_id TEXT,
                author_display_name TEXT,
                author_profile_url TEXT,
                author_json TEXT,
                raw_sample_json TEXT,
                updated_at TEXT,
                content_text TEXT
            );
            CREATE TABLE web_post_images(
                id INTEGER PRIMARY KEY,
                web_post_id INTEGER NOT NULL REFERENCES web_posts(id),
                image_url TEXT NOT NULL
            );
            INSERT INTO source_platforms VALUES('weibo', '微博');
            INSERT INTO source_platforms VALUES('douyin', '抖音');
            INSERT INTO source_platforms VALUES('zhihu', '知乎');
            INSERT INTO source_platforms VALUES('xhs', '小红书');
            """
        )
        conn.executemany("INSERT INTO web_posts VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.executemany(
            "INSERT INTO web_post_images VALUES(?, ?, ?)",
            [(1, 1, "keep-image")] if images is None else images,
        )
        if waivers is not None:
            conn.execute(
                """
                CREATE TABLE post_detail_repair_waivers(
                    id INTEGER PRIMARY KEY,
                    web_post_id INTEGER NOT NULL REFERENCES web_posts(id),
                    reason TEXT NOT NULL
                )
                """
            )
            conn.executemany("INSERT INTO post_detail_repair_waivers VALUES(?, ?, ?)", waivers)


def _source_rows() -> list[tuple[object, ...]]:
    weibo_id = "weibo-user-1001"
    short_id = "weibo-user-short"
    absent_id = "weibo-user-absent"
    douyin_id = SECRET_UID
    masked_id = "douyin-user-masked"
    zhihu_id = "zhihu-user-3003"
    return [
        (1, "weibo", "post-weibo-kept", _digest(weibo_id), _mask("张小三"), "https://example.test/weibo",
         _author(_digest(weibo_id), _mask("张小三")), _raw(_digest(weibo_id), platform="weibo", nickname=_mask("张小三"), source="source-snapshot"),
         "SOURCE-KEPT-WEIBO", "来源正文甲"),
        (2, "weibo", "post-weibo-short", _digest(short_id), _mask("李四"), "https://example.test/short",
         _author(_digest(short_id), _mask("李四")), _raw(_digest(short_id), platform="weibo", nickname=_mask("李四"), source="source-short"),
         "SOURCE-KEPT-SHORT", "来源短昵称"),
        (3, "weibo", "post-weibo-absent", _digest(absent_id), _mask("缺席人"), None,
         _author(_digest(absent_id), _mask("缺席人")), _raw(_digest(absent_id), platform="weibo", nickname=_mask("缺席人"), source="source-absent"),
         "SOURCE-KEPT-ABSENT", "无人认领"),
        (4, "douyin", "post-douyin-renamed", _digest(douyin_id), "李***四", "https://example.test/douyin",
         _author(_digest(douyin_id), "李***四"), _raw(_digest(douyin_id), platform="douyin", nickname="李***四", source="source-douyin"),
         "SOURCE-KEPT-DOUYIN", "来源正文乙"),
        (5, "douyin", "post-douyin-masked", _digest(masked_id), "*", "https://example.test/masked",
         _author(_digest(masked_id), "*"), _raw(_digest(masked_id), platform="douyin", nickname="*", source="source-masked"),
         "SOURCE-KEPT-MASKED", "仍脱敏"),
        (6, "zhihu", "post-zhihu", _digest(zhihu_id), _mask("知乎用户甲"), None,
         _author(_digest(zhihu_id), _mask("知乎用户甲")),
         _raw(_digest(zhihu_id), platform="zhihu", nickname=_mask("知乎用户甲"), user_nickname=_mask("知乎用户甲"), source="source-zhihu"),
         "SOURCE-KEPT-ZHIHU", "来源正文丙"),
        (7, "xhs", "post-xhs", "xhs-author", "小红书昵称", "https://example.test/xhs",
         _author("xhs-author", "小红书昵称"), _raw("xhs-author", platform="xhs", nickname="小红书昵称", source="source-xhs"),
         "SOURCE-KEPT-XHS", "其他平台不动"),
    ]


def _identity_rows() -> list[tuple[object, ...]]:
    weibo_id = "weibo-user-1001"
    short_id = "weibo-user-short"
    masked_id_hash = _digest("douyin-user-masked")
    zhihu_id = "zhihu-user-3003"
    return [
        (1, "weibo", "post-weibo-kept", weibo_id, "张小三", "https://identity.example/weibo",
         _author(weibo_id, "张小三", followers_count=99, profile={"city": "北京"}),
         _raw(weibo_id, platform="weibo", nickname="张小三", source="identity-live"),
         "IDENTITY-TOUCHED", "身份库正文甲"),
        (2, "weibo", "post-weibo-short", short_id, "李四", "https://identity.example/short",
         _author(short_id, "李四", followers_count=99, profile={"city": "北京"}),
         _raw(short_id, platform="weibo", nickname="李四", source="identity-short"),
         "IDENTITY-TOUCHED", "身份库短昵称"),
        (4, "douyin", "post-douyin-renamed", SECRET_UID, "王五五", "https://identity.example/douyin",
         _author(SECRET_UID, SECRET_NICK, followers_count=99, profile={"city": "北京"}),
         _raw(SECRET_UID, platform="douyin", nickname=SECRET_NICK, source="identity-douyin"),
         "IDENTITY-TOUCHED", "身份库正文乙"),
        (5, "douyin", "post-douyin-masked", masked_id_hash, "*", "https://identity.example/masked",
         _author(masked_id_hash, "*", followers_count=99, profile={"city": "北京"}),
         _raw(masked_id_hash, platform="douyin", nickname="*", source="identity-masked"),
         "IDENTITY-TOUCHED", "身份库仍是旧值"),
        (6, "zhihu", "post-zhihu", zhihu_id, "知乎用户甲", None,
         _author(zhihu_id, SECRET_NICK, followers_count=99, profile={"city": "北京"}),
         json.dumps({
             "content_detail_source": "identity-zhihu",
             "creator_hash": zhihu_id,
             "identity_only": 1,
             "nickname": SECRET_NICK,
             "user_nickname": "知乎用户甲",
         }, ensure_ascii=False, sort_keys=True),
         "IDENTITY-TOUCHED", "身份库正文丙"),
        (7, "xhs", "post-xhs", "identity-xhs", "不该覆盖", "https://identity.example/xhs",
         _author("identity-xhs", "不该覆盖"), _raw("identity-xhs", platform="xhs", nickname="不该覆盖", source="identity-xhs"),
         "IDENTITY-TOUCHED", "身份库其他平台"),
    ]


def _fixtures(directory: Path) -> tuple[Path, Path, bytes, bytes]:
    source = directory / "source.sqlite"
    identity = directory / "identity.sqlite"
    _create_database(source, _source_rows())
    _create_database(identity, _identity_rows())
    os.chmod(source, 0o444)
    os.chmod(identity, 0o400)
    return source, identity, source.read_bytes(), identity.read_bytes()


def _derive(directory: Path, source: Path, identity: Path, *, drop_unresolved: bool = False) -> dict[str, object]:
    return derive_author_identity_revision(
        source,
        identity,
        directory / "derived.sqlite",
        directory / "manifest.json",
        revision_id="rev-test-1",
        reason="synthetic-author-revision",
        code_commit="test-commit",
        drop_unresolved=drop_unresolved,
    )


def _rich_fixtures(directory: Path) -> tuple[Path, Path, bytes, bytes]:
    """在默认合成库上补被删帖的图片和豁免，用来区分保留行与关联行。"""

    source = directory / "source.sqlite"
    identity = directory / "identity.sqlite"
    _create_database(
        source,
        _source_rows(),
        images=[
            (1, 1, "keep-image"),
            (2, 3, "image-of-absent"),
            (3, 5, "image-of-masked"),
            (4, 6, "keep-zhihu-image"),
        ],
        waivers=[
            (1, 3, "waiver-of-absent-post"),
            (2, 6, "waiver-of-kept-post"),
        ],
    )
    _create_database(identity, _identity_rows())
    os.chmod(source, 0o444)
    os.chmod(identity, 0o400)
    return source, identity, source.read_bytes(), identity.read_bytes()


def test_derives_masked_authors_without_inheriting_source_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """只读来源仍可复制；输出权限是发布时设置的 0400，不是来源的 0444。"""

    def reject_metadata_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("复制不得继承权限或文件标志")

    monkeypatch.setattr(shutil, "copy2", reject_metadata_copy)
    monkeypatch.setattr(shutil, "copystat", reject_metadata_copy)
    source, identity, source_bytes, identity_bytes = _fixtures(tmp_path)
    source_mode = source.stat().st_mode
    source_flags = getattr(source.stat(), "st_flags", 0)
    result = _derive(tmp_path, source, identity)
    output = tmp_path / "derived.sqlite"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    unresolved = json.loads((tmp_path / "unresolved-posts.json").read_text())

    assert source.read_bytes() == source_bytes
    assert identity.read_bytes() == identity_bytes
    assert source.stat().st_mode == source_mode
    assert getattr(source.stat(), "st_flags", 0) == source_flags
    assert (output.stat().st_mode & 0o777) == 0o400
    assert (output.stat().st_mode & 0o777) != (source_mode & 0o777)
    assert getattr(output.stat(), "st_flags", 0) & stat.UF_IMMUTABLE == 0
    assert not any(Path(str(output) + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))
    assert result["status"] == "DERIVED_NOT_FROZEN"
    assert manifest["platforms"] == {
        "douyin": {"absent": 0, "renamed": 1, "replaced": 1, "rows": 2, "still_masked": 1},
        "weibo": {"absent": 1, "renamed": 0, "replaced": 2, "rows": 3, "still_masked": 0},
        "zhihu": {"absent": 0, "renamed": 0, "replaced": 1, "rows": 1, "still_masked": 0},
    }
    assert manifest["validation"]["quick_check"] == "ok"
    assert manifest["validation"]["changed_row_rehash"] == "passed"
    manifest_text = (tmp_path / "manifest.json").read_text()
    unresolved_text = (tmp_path / "unresolved-posts.json").read_text()
    for secret in (SECRET_NICK, SECRET_UID, "张小三", "王五五", "知乎用户甲"):
        assert secret not in manifest_text
        assert secret not in unresolved_text
    assert "drop_unresolved" not in manifest
    assert "dropped_posts" not in manifest
    assert unresolved["absent"] == [
        {"platform_key": "weibo", "platform_post_id": "post-weibo-absent", "source_post_id": 3}
    ]
    assert unresolved["still_masked"] == [
        {"platform_key": "douyin", "platform_post_id": "post-douyin-masked", "source_post_id": 5}
    ]

    with sqlite3.connect(output.as_uri() + "?mode=ro", uri=True) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "delete"
        rows = {row[0]: row for row in conn.execute(
            "SELECT platform_post_id, author_platform_id, author_display_name, author_profile_url, "
            "author_json, raw_sample_json, updated_at, content_text FROM web_posts"
        )}
        assert rows["post-weibo-kept"][1] == "weibo-user-1001"
        assert rows["post-weibo-kept"][2] == "张小三"
        assert rows["post-weibo-short"][1] == "weibo-user-short"
        assert rows["post-weibo-short"][2] == "李四"
        assert rows["post-weibo-absent"][1] == _digest("weibo-user-absent")
        assert rows["post-weibo-absent"][2] == _mask("缺席人")
        assert rows["post-douyin-masked"][1] == _digest("douyin-user-masked")
        assert rows["post-douyin-masked"][2] == "*"
        assert rows["post-xhs"][1:] == (
            "xhs-author", "小红书昵称", "https://example.test/xhs",
            _author("xhs-author", "小红书昵称"),
            _raw("xhs-author", platform="xhs", nickname="小红书昵称", source="source-xhs"),
            "SOURCE-KEPT-XHS", "其他平台不动",
        )
        for post_id, updated_at in (
            ("post-weibo-kept", "SOURCE-KEPT-WEIBO"),
            ("post-weibo-short", "SOURCE-KEPT-SHORT"),
            ("post-zhihu", "SOURCE-KEPT-ZHIHU"),
            ("post-douyin-renamed", "SOURCE-KEPT-DOUYIN"),
        ):
            assert rows[post_id][6] == updated_at
        assert rows["post-weibo-kept"][3] == "https://example.test/weibo"
        assert rows["post-zhihu"][3] is None
        assert rows["post-weibo-absent"][3] is None
        kept_author = json.loads(rows["post-weibo-kept"][4])
        assert kept_author["creator_hash"] == "weibo-user-1001"
        assert kept_author["nickname"] == "张小三"
        assert kept_author["followers_count"] == 12
        assert kept_author["profile"] == {"city": "青岛"}
        kept_raw = json.loads(rows["post-weibo-kept"][5])
        assert kept_raw["content_detail_source"] == "source-snapshot"
        assert kept_raw["extra_flag"] is True
        assert "identity_only" not in kept_raw
        zhihu_raw = json.loads(rows["post-zhihu"][5])
        assert zhihu_raw["creator_hash"] == "zhihu-user-3003"
        assert zhihu_raw["user_nickname"] == "知乎用户甲"
        assert "nickname" not in zhihu_raw
        assert zhihu_raw["content_detail_source"] == "source-zhihu"
        assert zhihu_raw["extra_flag"] is True
        assert "identity_only" not in zhihu_raw
        zhihu_author = json.loads(rows["post-zhihu"][4])
        assert zhihu_author["nickname"] == SECRET_NICK
        assert zhihu_author["profile"]["city"] == "青岛"
        assert conn.execute("SELECT image_url FROM web_post_images").fetchall() == [("keep-image",)]
        assert conn.execute("SELECT label FROM source_platforms WHERE platform_key='weibo'").fetchone()[0] == "微博"


def test_hash_mismatch_removes_temporary_and_writes_nothing(tmp_path: Path) -> None:
    source, identity, source_bytes, _identity_bytes = _fixtures(tmp_path)
    os.chmod(identity, 0o600)
    with sqlite3.connect(identity) as conn:
        conn.execute(
            "UPDATE web_posts SET author_platform_id=? WHERE platform_post_id='post-weibo-kept'",
            ("NOT_THE_PREIMAGE",),
        )
    output = tmp_path / "derived.sqlite"
    with pytest.raises(ValueError, match="author_identity_hash_mismatch"):
        _derive(tmp_path, source, identity)
    assert source.read_bytes() == source_bytes
    assert not output.exists()
    assert not Path(str(output) + ".tmp").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "unresolved-posts.json").exists()


def test_existing_output_is_rejected(tmp_path: Path) -> None:
    source, identity, source_bytes, _identity_bytes = _fixtures(tmp_path)
    output = tmp_path / "derived.sqlite"
    output.write_bytes(b"keep-existing-output")
    with pytest.raises(FileExistsError, match="author_identity_output_exists"):
        _derive(tmp_path, source, identity)
    assert output.read_bytes() == b"keep-existing-output"
    assert source.read_bytes() == source_bytes
    assert not Path(str(output) + ".tmp").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_source_wal_or_shm_is_rejected(tmp_path: Path, suffix: str) -> None:
    source, identity, source_bytes, _identity_bytes = _fixtures(tmp_path)
    sidecar = Path(str(source) + suffix)
    sidecar.write_bytes(b"")
    output = tmp_path / "derived.sqlite"
    with pytest.raises(ValueError, match="author_identity_source_sidecar"):
        _derive(tmp_path, source, identity)
    assert source.read_bytes() == source_bytes
    assert sidecar.exists()
    assert not output.exists()


def test_script_writes_commit_and_derived_status(tmp_path: Path) -> None:
    source, identity, _source_bytes, _identity_bytes = _fixtures(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "derived.sqlite"
    manifest = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            str(PYTHON),
            str(repo / "scripts/cleaning_derive_author_identity_revision.py"),
            "--source", str(source),
            "--identity-db", str(identity),
            "--output", str(output),
            "--manifest", str(manifest),
            "--revision-id", "rev-cli",
            "--reason", "synthetic-cli",
        ],
        check=True,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    printed = json.loads(completed.stdout)
    stored = json.loads(manifest.read_text())
    assert printed["code_commit"] == head
    assert stored["status"] == "DERIVED_NOT_FROZEN"
    assert SECRET_NICK not in manifest.read_text()
    assert SECRET_UID not in completed.stdout


def test_without_drop_unresolved_keeps_masked_posts_and_related_rows(tmp_path: Path) -> None:
    """不开启删除时，未恢复帖、图片和豁免都留在副本里，来源字节与权限不变。"""

    source, identity, source_bytes, identity_bytes = _rich_fixtures(tmp_path)
    source_mode = source.stat().st_mode
    source_flags = getattr(source.stat(), "st_flags", 0)
    result = _derive(tmp_path, source, identity, drop_unresolved=False)
    output = tmp_path / "derived.sqlite"

    assert source.read_bytes() == source_bytes
    assert identity.read_bytes() == identity_bytes
    assert source.stat().st_mode == source_mode
    assert getattr(source.stat(), "st_flags", 0) == source_flags
    assert "drop_unresolved" not in result
    assert "dropped_posts" not in result
    with sqlite3.connect(output.as_uri() + "?mode=ro", uri=True) as conn:
        assert [row[0] for row in conn.execute("SELECT id FROM web_posts ORDER BY id")] == [1, 2, 3, 4, 5, 6, 7]
        absent = conn.execute(
            "SELECT author_platform_id, author_display_name, content_text FROM web_posts WHERE id=3"
        ).fetchone()
        assert absent == (_digest("weibo-user-absent"), _mask("缺席人"), "无人认领")
        masked = conn.execute(
            "SELECT author_platform_id, author_display_name FROM web_posts WHERE id=5"
        ).fetchone()
        assert masked == (_digest("douyin-user-masked"), "*")
        assert conn.execute("SELECT id, web_post_id, image_url FROM web_post_images ORDER BY id").fetchall() == [
            (1, 1, "keep-image"),
            (2, 3, "image-of-absent"),
            (3, 5, "image-of-masked"),
            (4, 6, "keep-zhihu-image"),
        ]
        assert conn.execute(
            "SELECT id, web_post_id, reason FROM post_detail_repair_waivers ORDER BY id"
        ).fetchall() == [
            (1, 3, "waiver-of-absent-post"),
            (2, 6, "waiver-of-kept-post"),
        ]


def test_drop_unresolved_removes_related_rows_and_vacuums_deleted_text(tmp_path: Path) -> None:
    """开启后删除未恢复帖及其关联行，保留行的 id 与非作者列不变，空闲页不再含被删正文。"""

    source, identity, source_bytes, identity_bytes = _rich_fixtures(tmp_path)
    source_mode = source.stat().st_mode
    source_flags = getattr(source.stat(), "st_flags", 0)
    result = _derive(tmp_path, source, identity, drop_unresolved=True)
    output = tmp_path / "derived.sqlite"
    manifest_text = (tmp_path / "manifest.json").read_text()
    unresolved_text = (tmp_path / "unresolved-posts.json").read_text()
    unresolved = json.loads(unresolved_text)

    assert source.read_bytes() == source_bytes
    assert identity.read_bytes() == identity_bytes
    assert source.stat().st_mode == source_mode
    assert getattr(source.stat(), "st_flags", 0) == source_flags
    assert (output.stat().st_mode & 0o777) == 0o400
    assert getattr(output.stat(), "st_flags", 0) & stat.UF_IMMUTABLE == 0
    assert not any(Path(str(output) + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))
    assert result["drop_unresolved"] is True
    assert result["dropped_posts"] == 2
    assert result["dropped_images"] == 2
    assert result["dropped_waivers"] == 1
    assert result["validation"]["vacuum"] == "completed"
    assert result["validation"]["web_posts_row_count"] == "source_minus_dropped"
    assert result["validation"]["unresolved_masked_rows"] == "none"
    assert result["platforms"]["weibo"]["absent"] == 1
    assert result["platforms"]["douyin"]["still_masked"] == 1
    for secret in (SECRET_NICK, SECRET_UID, "张小三", "王五五"):
        assert secret not in manifest_text
        assert secret not in unresolved_text
    assert unresolved["absent"] == [
        {"platform_key": "weibo", "platform_post_id": "post-weibo-absent", "source_post_id": 3}
    ]
    assert unresolved["still_masked"] == [
        {"platform_key": "douyin", "platform_post_id": "post-douyin-masked", "source_post_id": 5}
    ]

    blob = output.read_bytes()
    for removed in ("无人认领", "仍脱敏", "image-of-absent", "image-of-masked", "waiver-of-absent-post"):
        assert removed.encode() not in blob
    assert "来源正文甲".encode() in blob
    assert "waiver-of-kept-post".encode() in blob

    with sqlite3.connect(output.as_uri() + "?mode=ro", uri=True) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert [row[0] for row in conn.execute("SELECT id FROM web_posts ORDER BY id")] == [1, 2, 4, 6, 7]
        kept = {
            row[0]: row
            for row in conn.execute(
                "SELECT id, platform_post_id, author_platform_id, author_display_name, author_profile_url, "
                "updated_at, content_text FROM web_posts"
            )
        }
        assert kept[1][2:] == ("weibo-user-1001", "张小三", "https://example.test/weibo", "SOURCE-KEPT-WEIBO", "来源正文甲")
        assert kept[7][1:] == (
            "post-xhs",
            "xhs-author",
            "小红书昵称",
            "https://example.test/xhs",
            "SOURCE-KEPT-XHS",
            "其他平台不动",
        )
        assert conn.execute("SELECT id, web_post_id, image_url FROM web_post_images ORDER BY id").fetchall() == [
            (1, 1, "keep-image"),
            (4, 6, "keep-zhihu-image"),
        ]
        assert conn.execute("SELECT id, web_post_id, reason FROM post_detail_repair_waivers").fetchall() == [
            (2, 6, "waiver-of-kept-post"),
        ]
        masked_rows = conn.execute(
            """
            SELECT platform_key, author_platform_id, author_display_name
            FROM web_posts
            WHERE platform_key IN ('weibo', 'douyin', 'zhihu')
            """
        ).fetchall()
        assert masked_rows
        for platform, author_id, display_name in masked_rows:
            assert not (isinstance(author_id, str) and len(author_id) == 16 and _mask_like(display_name))


def _mask_like(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value == "*":
        return True
    if len(value) == 2 and value[1] == "*":
        return True
    return len(value) == 5 and value[1:4] == "***"
