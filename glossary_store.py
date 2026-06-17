"""
Glossary DÙNG CHUNG (global) — kho SQLite tham chiếu cho MỌI bộ truyện.

Khác với glossary.txt per-bộ (nằm ở gốc từng bộ, custom_openai đọc qua
MIT_GLOSSARY_PATH): file này là MỘT kho duy nhất ở gốc repo, mọi bộ có thể
opt-in tham chiếu khi dịch. Mỗi mục là một cặp khoá cứng `zh => vi` (giống
per-bộ) nhưng thêm:
  - enabled : 1 = đã duyệt, được áp khi dịch; 0 = chờ duyệt (tự học đẩy lên).
  - sources : JSON list tên bộ đã đóng góp / gặp mục này (chỉ để tham khảo trên UI).

Khoá UNIQUE(zh) → tự dedup theo chữ Hán gốc: thêm trùng không nhân đôi.
Module CHỈ dùng stdlib (sqlite3) nên chạy được cả trong mit_venv (custom_openai).

Dùng bởi:
  - web_app.py            → API quản lý (list/sửa/bật-tắt/xoá/dedup).
  - _mit_backend.py       → lấy db_path() truyền env MIT_GLOSSARY_DB cho subprocess.
  - custom_openai (patch) → load_enabled_terms() khi dịch + learn_pending() khi tự học.
"""

from __future__ import annotations

import json
import sqlite3
import time
import unicodedata
from pathlib import Path

# Kho mặc định: cùng thư mục với module này (gốc repo). web_app & _mit_backend
# chạy trong process có repo root trên sys.path nên import thẳng được; custom_openai
# (subprocess) nhận đường dẫn tuyệt đối qua env MIT_GLOSSARY_DB.
_DEFAULT_DB = Path(__file__).resolve().parent / "glossary_global.db"


def db_path() -> str:
    """Đường dẫn tuyệt đối tới kho global dùng chung (cùng chỗ cho mọi caller)."""
    return str(_DEFAULT_DB)


def _norm(zh: str) -> str:
    """Chuẩn hoá chữ Hán để so trùng: bỏ khoảng trắng đầu/cuối + NFC unicode."""
    return unicodedata.normalize("NFC", (zh or "").strip())


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _connect(path: str | None = None) -> sqlite3.Connection:
    """Mở kết nối + bảo đảm schema. WAL + busy_timeout để nhiều tiến trình dịch
    (mỗi bộ một subprocess) ghi song song không khoá nhau."""
    p = path or db_path()
    parent = Path(p).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS terms (
            id      INTEGER PRIMARY KEY,
            zh      TEXT NOT NULL,
            vi      TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            sources TEXT,
            created TEXT,
            updated TEXT
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_terms_zh ON terms(zh)")
    conn.commit()
    return conn


def _merge_sources(existing: str | None, source: str | None) -> str:
    """Gộp 1 tên bộ vào JSON list sources (không trùng), giữ thứ tự xuất hiện."""
    try:
        lst = json.loads(existing) if existing else []
        if not isinstance(lst, list):
            lst = []
    except Exception:
        lst = []
    if source and source not in lst:
        lst.append(source)
    return json.dumps(lst, ensure_ascii=False)


# ── Dùng khi DỊCH (custom_openai) ────────────────────────────────────────────
def load_enabled_terms(path: str | None = None) -> list[tuple[str, str]]:
    """Trả list[(zh, vi)] các mục ĐÃ DUYỆT (enabled=1), sort DÀI→NGẮN để replace
    cụm dài trước (青云宗 trước 青云). Lỗi/không có DB → []."""
    try:
        conn = _connect(path)
        try:
            rows = conn.execute(
                "SELECT zh, vi FROM terms WHERE enabled=1"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []
    terms = [(r["zh"], r["vi"]) for r in rows if r["zh"] and r["vi"]]
    terms.sort(key=lambda t: len(t[0]), reverse=True)
    return terms


def learn_pending(path: str | None, pairs, source: str | None = None) -> int:
    """Tự học → đẩy các cặp MỚI lên global ở trạng thái CHỜ DUYỆT (enabled=0).
    Chỉ thêm zh CHƯA có (INSERT OR IGNORE) → KHÔNG đụng mục người dùng đã duyệt/sửa.
    Với mục đã tồn tại: chỉ gộp thêm tên bộ vào sources. Trả số mục mới thêm."""
    rows = [(_norm(zh), (vi or "").strip()) for zh, vi in (pairs or [])]
    rows = [(zh, vi) for zh, vi in rows if zh and vi]
    if not rows:
        return 0
    added = 0
    try:
        conn = _connect(path)
        try:
            for zh, vi in rows:
                src = _merge_sources(None, source)
                cur = conn.execute(
                    "INSERT OR IGNORE INTO terms(zh, vi, enabled, sources, created, updated) "
                    "VALUES (?,?,0,?,?,?)",
                    (zh, vi, src, _now(), _now()),
                )
                if cur.rowcount:
                    added += 1
                elif source:
                    # đã có → chỉ ghi nhận bộ này cũng gặp mục đó
                    row = conn.execute(
                        "SELECT sources FROM terms WHERE zh=?", (zh,)
                    ).fetchone()
                    conn.execute(
                        "UPDATE terms SET sources=? WHERE zh=?",
                        (_merge_sources(row["sources"] if row else None, source), zh),
                    )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        return added
    return added


# ── Dùng bởi UI / API (web_app) ──────────────────────────────────────────────
def _row_to_dict(r: sqlite3.Row) -> dict:
    try:
        sources = json.loads(r["sources"]) if r["sources"] else []
    except Exception:
        sources = []
    return {
        "id": r["id"],
        "zh": r["zh"],
        "vi": r["vi"],
        "enabled": bool(r["enabled"]),
        "sources": sources,
        "updated": r["updated"],
    }


def list_all(path: str | None = None, q: str | None = None,
             status: str | None = None) -> list[dict]:
    """Liệt kê cho UI. q = lọc theo chuỗi con trong zh/vi (không phân biệt hoa/thường);
    status ∈ {'enabled','pending', None}. Sắp xếp: chờ duyệt lên đầu, rồi mới nhất."""
    where, params = [], []
    if q:
        where.append("(zh LIKE ? OR vi LIKE ?)")
        like = f"%{q}%"
        params += [like, like]
    if status == "enabled":
        where.append("enabled=1")
    elif status == "pending":
        where.append("enabled=0")
    sql = "SELECT * FROM terms"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY enabled ASC, COALESCE(updated, created) DESC, id DESC"
    conn = _connect(path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def counts(path: str | None = None) -> dict:
    """{total, enabled, pending} để hiển thị nhanh trên UI."""
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN enabled=1 THEN 1 ELSE 0 END) AS enabled FROM terms"
        ).fetchone()
    finally:
        conn.close()
    total = row["total"] or 0
    enabled = row["enabled"] or 0
    return {"total": total, "enabled": enabled, "pending": total - enabled}


def upsert(path: str | None, zh: str, vi: str, enabled: bool = True,
           source: str | None = None) -> dict:
    """Thêm cặp mới HOẶC cập nhật theo zh (khoá UNIQUE → tự dedup). Trả mục sau ghi."""
    zh, vi = _norm(zh), (vi or "").strip()
    if not zh or not vi:
        raise ValueError("zh và vi không được rỗng")
    conn = _connect(path)
    try:
        row = conn.execute("SELECT * FROM terms WHERE zh=?", (zh,)).fetchone()
        if row:
            conn.execute(
                "UPDATE terms SET vi=?, enabled=?, sources=?, updated=? WHERE id=?",
                (vi, 1 if enabled else 0,
                 _merge_sources(row["sources"], source), _now(), row["id"]),
            )
            new_id = row["id"]
        else:
            cur = conn.execute(
                "INSERT INTO terms(zh, vi, enabled, sources, created, updated) "
                "VALUES (?,?,?,?,?,?)",
                (zh, vi, 1 if enabled else 0,
                 _merge_sources(None, source), _now(), _now()),
            )
            new_id = cur.lastrowid
        conn.commit()
        out = conn.execute("SELECT * FROM terms WHERE id=?", (new_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_dict(out)


def update_pair(path: str | None, term_id: int, zh: str, vi: str) -> dict:
    """Sửa cả zh lẫn vi của một mục (UI sửa inline). Nếu zh mới trùng mục khác →
    lỗi UNIQUE → gộp: chuyển vi sang mục đang giữ zh đó, xoá mục này."""
    zh, vi = _norm(zh), (vi or "").strip()
    if not zh or not vi:
        raise ValueError("zh và vi không được rỗng")
    conn = _connect(path)
    try:
        clash = conn.execute(
            "SELECT id FROM terms WHERE zh=? AND id<>?", (zh, term_id)
        ).fetchone()
        if clash:
            # zh mới đụng mục khác → cập nhật mục đó, bỏ mục hiện tại (dedup khi sửa)
            conn.execute("UPDATE terms SET vi=?, updated=? WHERE id=?",
                         (vi, _now(), clash["id"]))
            conn.execute("DELETE FROM terms WHERE id=?", (term_id,))
            keep_id = clash["id"]
        else:
            conn.execute("UPDATE terms SET zh=?, vi=?, updated=? WHERE id=?",
                         (zh, vi, _now(), term_id))
            keep_id = term_id
        conn.commit()
        out = conn.execute("SELECT * FROM terms WHERE id=?", (keep_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_dict(out) if out else {}


def set_enabled(path: str | None, ids, enabled: bool) -> int:
    """Bật/tắt hàng loạt theo list id. Trả số dòng đổi."""
    ids = [int(i) for i in (ids or [])]
    if not ids:
        return 0
    conn = _connect(path)
    try:
        ph = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"UPDATE terms SET enabled=?, updated=? WHERE id IN ({ph})",
            [1 if enabled else 0, _now(), *ids],
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete(path: str | None, ids) -> int:
    """Xoá hàng loạt theo list id. Trả số dòng xoá."""
    ids = [int(i) for i in (ids or [])]
    if not ids:
        return 0
    conn = _connect(path)
    try:
        ph = ",".join("?" for _ in ids)
        cur = conn.execute(f"DELETE FROM terms WHERE id IN ({ph})", ids)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def dedup(path: str | None = None) -> dict:
    """Gộp các mục có zh trùng nhau SAU chuẩn hoá (NFC + strip). zh trong DB lẽ ra
    đã unique, nhưng dữ liệu nhập tay / khác cách gõ unicode có thể tạo biến thể.
    Giữ mục enabled (ưu tiên) hoặc cũ nhất làm chuẩn, gộp sources, xoá phần dư.
    Trả {merged, total}."""
    conn = _connect(path)
    merged = 0
    try:
        rows = conn.execute(
            "SELECT * FROM terms ORDER BY enabled DESC, id ASC"
        ).fetchall()
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(_norm(r["zh"]), []).append(r)
        for norm_zh, grp in groups.items():
            if len(grp) <= 1:
                # vẫn chuẩn hoá zh của mục đơn (phòng zh chưa normalize)
                if grp and grp[0]["zh"] != norm_zh:
                    conn.execute("UPDATE terms SET zh=?, updated=? WHERE id=?",
                                 (norm_zh, _now(), grp[0]["id"]))
                continue
            keep = grp[0]  # đã sort enabled DESC, id ASC → mục tốt nhất đứng đầu
            srcs = keep["sources"]
            for dup in grp[1:]:
                # gộp sources của bản trùng vào bản giữ lại
                try:
                    for s in (json.loads(dup["sources"]) if dup["sources"] else []):
                        srcs = _merge_sources(srcs, s)
                except Exception:
                    pass
                conn.execute("DELETE FROM terms WHERE id=?", (dup["id"],))
                merged += 1
            conn.execute("UPDATE terms SET zh=?, sources=?, updated=? WHERE id=?",
                         (norm_zh, srcs, _now(), keep["id"]))
        conn.commit()
        total = conn.execute("SELECT COUNT(*) AS c FROM terms").fetchone()["c"]
    finally:
        conn.close()
    return {"merged": merged, "total": total}


def promote_terms(path: str | None, pairs, source: str | None = None,
                  enabled: bool = True) -> int:
    """Đưa các cặp (zh, vi) từ glossary per-bộ lên global, ENABLE luôn (user chủ
    động bấm 'đưa bộ này lên chuẩn chung'). zh đã có → cập nhật vi + enable + gộp
    source. Trả số mục thêm/sửa."""
    n = 0
    for zh, vi in (pairs or []):
        try:
            upsert(path, zh, vi, enabled=enabled, source=source)
            n += 1
        except Exception:
            continue
    return n
