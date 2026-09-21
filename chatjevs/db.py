"""Local SQLite store: API key, response cache, runs, generated steps, evals."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

DEFAULT = os.path.join(os.getcwd(), "chatjevs.db")
PATH = os.environ.get("CHATJEVS_DB", DEFAULT)
LEGACY = os.path.join(os.getcwd(), "jevnet.db")   # the project's former name

SCHEMA = """
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS cache  (hash TEXT PRIMARY KEY, response TEXT, created REAL);
CREATE TABLE IF NOT EXISTS vocabs (name TEXT PRIMARY KEY, words TEXT, created REAL);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, arch TEXT, prompt TEXT, output TEXT,
  jevs INTEGER, requests INTEGER, cached INTEGER, in_tokens INTEGER, out_tokens INTEGER,
  seconds REAL, meta TEXT, created REAL);
CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, idx INTEGER,
  token TEXT, state TEXT, dist TEXT);
CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, arch TEXT, task TEXT,
  category TEXT, chosen TEXT, answer TEXT, correct INTEGER, probs TEXT);
"""


WRITE = threading.Lock()   # one connection is shared across request threads


def connect(path: str | None = None) -> sqlite3.Connection:
    target = path or PATH
    # only ever adopt the old file in place of the default one, never a path
    # the caller chose explicitly
    if target == DEFAULT and not os.path.exists(target) and os.path.exists(LEGACY):
        try:
            os.rename(LEGACY, target)   # keeps the stored API key across the rename
            print(f"moved {LEGACY} -> {target}")
        except OSError:                 # open elsewhere; keep using it rather than fail
            target = LEGACY
            print(f"note: using {LEGACY}; rename it once nothing else has it open")
    con = sqlite3.connect(target, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def get(con, key: str, default=None):
    row = con.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def _write(con, sql: str, args=()):
    with WRITE:
        cur = con.execute(sql, args)
        con.commit()
        return cur


def put(con, key: str, value: str) -> None:
    _write(con, "INSERT INTO config(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def cache_get(con, h: str):
    with WRITE:
        row = con.execute("SELECT response FROM cache WHERE hash=?", (h,)).fetchone()
    return json.loads(row["response"]) if row else None


def cache_put(con, h: str, response: dict) -> None:
    _write(con, "INSERT OR REPLACE INTO cache VALUES(?,?,?)",
           (h, json.dumps(response), time.time()))


def save_vocab(con, name: str, words) -> None:
    _write(con, "INSERT OR REPLACE INTO vocabs VALUES(?,?,?)",
           (name, json.dumps(words), time.time()))


def load_vocab(con, name: str):
    row = con.execute("SELECT words FROM vocabs WHERE name=?", (name,)).fetchone()
    return [tuple(x) for x in json.loads(row["words"])] if row else None


def start_run(con, kind: str, arch: str, prompt: str) -> int:
    return _write(con, "INSERT INTO runs(kind,arch,prompt,created) VALUES(?,?,?,?)",
                  (kind, arch, prompt, time.time())).lastrowid


def finish_run(con, run_id: int, output: str, usage: dict, seconds: float, meta: dict) -> None:
    _write(con, "UPDATE runs SET output=?,jevs=?,requests=?,cached=?,in_tokens=?,"
                "out_tokens=?,seconds=?,meta=? WHERE id=?",
           (output, meta.get("jevs", 0), usage.get("requests", 0), usage.get("cached", 0),
            usage.get("in_tokens", 0), usage.get("out_tokens", 0), seconds,
            json.dumps(meta), run_id))


def add_step(con, run_id: int, idx: int, token: str, state: dict, dist: dict) -> None:
    _write(con, "INSERT INTO steps(run_id,idx,token,state,dist) VALUES(?,?,?,?,?)",
           (run_id, idx, token, json.dumps(state), json.dumps(dist)))


def add_eval(con, run_id, arch, task, category, chosen, answer, probs) -> None:
    _write(con, "INSERT INTO evals(run_id,arch,task,category,chosen,answer,correct,probs) "
                "VALUES(?,?,?,?,?,?,?,?)",
           (run_id, arch, task, category, chosen, answer,
            int(chosen == answer), json.dumps(probs)))
