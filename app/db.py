import hashlib
import json
import re
import secrets
import sqlite3
import time

DB_PATH = "gateway.db"


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                config TEXT NOT NULL DEFAULT '{}'
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, tenant_id TEXT, model TEXT, status TEXT,
                latency_ms INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER
            )"""
        )


def log_request(tenant_id, model, status, latency_ms, prompt_tokens=0, completion_tokens=0):
    with _conn() as c:
        c.execute(
            "INSERT INTO requests (ts, tenant_id, model, status, latency_ms, "
            "prompt_tokens, completion_tokens) VALUES (?,?,?,?,?,?,?)",
            (time.time(), tenant_id, model, status, latency_ms,
             prompt_tokens, completion_tokens),
        )


def update_tenant_config(tenant_id, updates: dict):
    with _conn() as c:
        row = c.execute("SELECT config FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
        if row is None:
            raise ValueError("unknown tenant")
        cfg = json.loads(row["config"])
        cfg.update(updates)
        c.execute("UPDATE tenants SET config = ? WHERE tenant_id = ?",
                  (json.dumps(cfg), tenant_id))

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_tenant(tenant_id: str, name: str, config: dict | None = None) -> str:
    if not re.fullmatch(r"[a-z0-9_]+", tenant_id):
        raise ValueError("tenant_id must be lowercase letters, digits, underscores")
    key = "gw_" + secrets.token_urlsafe(32)
    with _conn() as c:
        c.execute(
            "INSERT INTO tenants VALUES (?, ?, ?, ?)",
            (tenant_id, name, hash_key(key), json.dumps(config or {})),
        )
    return key  # shown once; only the hash is stored


def get_tenant_by_key(key: str):
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM tenants WHERE key_hash = ?", (hash_key(key),)
        ).fetchone()
    if row is None:
        return None
    tenant = dict(row)
    tenant["config"] = json.loads(tenant["config"])
    return tenant