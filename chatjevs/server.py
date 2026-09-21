"""ChatJEVs: a local web UI. `python -m chatjevs serve`

Streams tokens over SSE as the network decides them, so a 4-request-per-word
model still feels like something is happening.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import bench, db, vocab as vocab_mod
from .client import Jev, JevError, noul
from .net import ARCHS, Net

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _num(value, default, cast=int):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def _vocab_for(con, arch):
    imported = db.load_vocab(con, db.get(con, "vocab", "builtin"))
    if imported and arch.vocab_size > 700:
        return vocab_mod.Vocab(imported[:arch.vocab_size])
    return vocab_mod.build(arch.vocab_size)


def _results(con):
    """Real benchmark rows, or an honest empty result. Nothing is invented."""
    rows = con.execute(
        "SELECT arch, category, COUNT(*) n, SUM(correct) hits FROM evals "
        "GROUP BY arch, category").fetchall()
    by_arch: dict[str, dict] = {}
    for r in rows:
        entry = by_arch.setdefault(r["arch"], {"by_category": {}, "n": 0, "hits": 0})
        entry["by_category"][r["category"]] = {"n": r["n"], "hits": r["hits"] or 0,
                                               "accuracy": (r["hits"] or 0) / r["n"]}
        entry["n"] += r["n"]
        entry["hits"] += r["hits"] or 0
    for entry in by_arch.values():
        entry["accuracy"] = entry["hits"] / entry["n"] if entry["n"] else 0.0
    gen = {}
    for r in con.execute("SELECT arch, meta FROM runs WHERE kind='bench' "
                         "AND meta IS NOT NULL ORDER BY id"):
        try:
            block = json.loads(r["meta"]).get("generation") or {}
        except ValueError:
            continue
        if block:
            gen[r["arch"]] = block
    cost = {r["arch"]: {"requests": r["requests"], "questions": r["q"], "seconds": r["s"]}
            for r in con.execute(
                "SELECT arch, SUM(requests) requests, SUM(jevs) q, SUM(seconds) s "
                "FROM runs WHERE kind='bench' GROUP BY arch")}
    chance = (sum(1 / len(t[3]) for t in bench.CLOZE) / len(bench.CLOZE)
              if bench.CLOZE else 0.0)
    return {"measured": bool(by_arch), "by_arch": by_arch, "cost": cost,
            "chance": chance, "generation": gen, "generation_checks": bench.CHECKS,
            "archs": [{"name": n, "jevs": a.jevs, "word_jevs": a.word_jevs,
                       "words": len(_vocab_for(con, a)), "note": a.note}
                      for n, a in ARCHS.items()],
            "categories": bench.CATEGORIES, "tasks": len(bench.CLOZE)}


def _recents(con, limit=25):
    rows = con.execute("SELECT id,kind,arch,prompt,output,created FROM runs "
                       "WHERE kind='gen' AND prompt IS NOT NULL "
                       "ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows]


class Handler(BaseHTTPRequestHandler):
    server_version = "ChatJEVs"
    protocol_version = "HTTP/1.1"

    # ---- plumbing --------------------------------------------------------
    def log_message(self, *args):
        pass

    def _json(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name):
        path = os.path.join(WEB, name)
        if not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        with open(path, "rb") as fh:
            body = fh.read()
        ctype = ("text/html; charset=utf-8" if name.endswith(".html")
                 else "application/javascript" if name.endswith(".js")
                 else "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")   # never serve a stale UI
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_open(self):
        self._streaming = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True    # SSE is unframed; the stream ends the response
        self._lock = threading.Lock()

    def _sse(self, event, data):
        chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
        with self._lock:
            self.wfile.write(chunk)
            self.wfile.flush()

    def _fail(self, exc):
        """Never let a handler die silently: the browser is waiting on us."""
        message = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        try:
            if getattr(self, "_streaming", False):
                self._sse("failed", {"message": message})
            else:
                self._json({"error": message}, 500)
        except Exception:
            pass

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    # ---- routes ----------------------------------------------------------
    def do_GET(self):
        path, _, raw = self.path.partition("?")
        q = {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}
        con = db.connect()
        try:
            if path in ("/", "/index.html"):
                return self._file("index.html")
            if path in ("/writeup", "/writeup.html"):
                return self._file("writeup.html")
            if path == "/brain.js":
                return self._file("brain.js")
            if path == "/api/results":
                return self._json(_results(con))
            if path == "/api/boot":
                return self._json({
                    "has_key": bool(db.get(con, "api_key") or os.environ.get("TYPESAFE_API_KEY")),
                    "archs": [{"name": n, "jevs": a.jevs, "note": a.note,
                               "words": len(_vocab_for(con, a)), "window": a.window,
                               "requests": a.requests_per_token, "plan": a.plan,
                               "semantic": a.semantic, "syntax": a.syntax,
                               "memory": a.memory, "concepts": a.concepts,
                               "routing": a.routing, "decoder": a.decoder,
                               "critics": a.critics, "drafts": a.drafts,
                               "word_jevs": a.word_jevs}
                              for n, a in ARCHS.items()],
                    "categories": bench.CATEGORIES,
                    "recents": _recents(con)})
            if path == "/api/run":
                run_id = _num(q.get("id"), 0)
                row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
                steps = con.execute("SELECT idx,token,state,dist FROM steps WHERE run_id=? "
                                    "ORDER BY idx", (run_id,))
                return self._json({
                    "run": dict(row) if row else None,
                    "steps": [{"idx": s["idx"], "token": s["token"],
                               "memory": json.loads(s["state"]).get("memory", {}),
                               "dist": json.loads(s["dist"])} for s in steps]})
            if path == "/api/chat":
                return self._chat(con, q)
            return self._json({"error": "not found"}, 404)
        except Exception as exc:
            return self._fail(exc)
        finally:
            con.close()

    def do_POST(self):
        con = db.connect()
        try:
            if self.path == "/api/key":
                key = (self._body().get("key") or "").strip()
                if not key:
                    return self._json({"ok": False, "error": "empty key"}, 400)
                try:
                    Jev(key, con, use_cache=False).ask(
                        "The cat sat on the mat.",
                        {"ok": noul("This text is written in English.")})
                except JevError as exc:
                    return self._json({"ok": False, "error": str(exc)}, 400)
                db.put(con, "api_key", key)
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)
        except Exception as exc:
            return self._fail(exc)
        finally:
            con.close()

    # ---- streams ---------------------------------------------------------
    def _make_jev(self, con, fresh=False):
        key = db.get(con, "api_key") or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise JevError("no API key stored")
        jev = Jev(key, con, use_cache=not fresh)
        jev.on_request = lambda label, n, cached: self._sse(
            "stage", {"label": label, "questions": n, "cached": cached})
        return jev

    def _chat(self, con, q):
        self._sse_open()
        arch = ARCHS.get(q.get("arch", "v1"), ARCHS["v1"])
        prompt = (q.get("prompt") or "").strip()
        started = time.time()
        try:
            jev = self._make_jev(con, q.get("fresh") == "1")
            vocab = _vocab_for(con, arch)
            net = Net(arch, jev, vocab, temperature=_num(q.get("temp"), 0.8, float))
            net.on_note = lambda kind, data: self._sse("note", {"kind": kind, "data": data})
            run_id = db.start_run(con, "gen", arch.name, prompt)
            self._sse("start", {"run_id": run_id, "jevs": arch.jevs, "words": len(vocab)})
            idx = 0

            def emit(word, dist, memory):
                nonlocal idx
                top = sorted(dist.items(), key=lambda kv: -kv[1])[:6]
                db.add_step(con, run_id, idx, word, {"memory": memory}, dist)
                self._sse("token", {"word": word, "memory": memory, "top": top})
                idx += 1

            net.generate(prompt, max(1, min(_num(q.get("n"), 12), 60)), on_token=emit)
            seconds = time.time() - started
            db.finish_run(con, run_id, " ".join(t["token"] for t in net.trace),
                          jev.usage, seconds, {"jevs": arch.jevs, "vocab": len(vocab)})
            self._sse("done", {"usage": jev.usage, "seconds": round(seconds, 1),
                               "memory": net.memory, "run_id": run_id})
        except JevError as exc:
            self._sse("failed", {"message": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._fail(exc)

def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"ChatJEVs on http://{host}:{port}  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
