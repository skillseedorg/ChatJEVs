"""Minimal TypeSafe System One (Jev) client: stdlib only, cached, batched."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request

from . import db

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_OPTIONS = 255          # Jev's per-Choice ceiling
RETRY_STATUS = {429, 500, 502, 503, 529}


class JevError(RuntimeError):
    pass


class Jev:
    """One ``ask`` = one batched request = N Jev units answered in parallel."""

    def __init__(self, api_key: str, con, model: str = "jev-latest",
                 use_cache: bool = True, batch: int = 64, workers: int = 8):
        self.api_key, self.con, self.model = api_key, con, model
        self.use_cache, self.batch, self.workers = use_cache, batch, workers
        self.on_request = None       # (label, n_questions, cached) -> None
        self.usage = {"requests": 0, "cached": 0, "in_tokens": 0, "out_tokens": 0, "questions": 0}
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def label(self) -> str:
        """What the current ask is for. Per-thread, since batches fan out."""
        return getattr(self._local, "label", "")

    @label.setter
    def label(self, value: str) -> None:
        self._local.label = value

    # ---- transport -------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                if exc.code in RETRY_STATUS and attempt < 4:
                    time.sleep((2 ** attempt) + random.random())
                    continue
                raise JevError(f"HTTP {exc.code}: {detail}") from None
            except urllib.error.URLError as exc:
                if attempt < 4:
                    time.sleep((2 ** attempt) + random.random())
                    continue
                raise JevError(f"connection failed: {exc.reason}") from None
        raise JevError("exhausted retries")

    def _one(self, state, questions: dict) -> dict:
        payload = {"state": state, "model": self.model, "questions": questions}
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if self.use_cache:
            with self._lock:
                hit = db.cache_get(self.con, h)
            if hit is not None:
                with self._lock:
                    self.usage["cached"] += 1
                    self.usage["questions"] += len(questions)
                self._notify(len(questions), True)
                return hit["answers"]
        data = self._post(payload)
        use = data.get("usage", {})
        with self._lock:
            if self.use_cache:
                db.cache_put(self.con, h, data)
            self.usage["requests"] += 1
            self.usage["questions"] += len(questions)
            self.usage["in_tokens"] += use.get("input_tokens", 0)
            self.usage["out_tokens"] += use.get("output_tokens", 0)
        self._notify(len(questions), False)
        return data.get("answers", {})

    def _notify(self, count: int, cached: bool) -> None:
        if self.on_request:
            try:
                self.on_request(self.label, count, cached)
            except Exception:
                pass

    # ---- public ----------------------------------------------------------
    def ask(self, state, questions: dict) -> dict:
        """Answer every question against one state, splitting oversized batches."""
        if not questions:
            return {}
        keys = list(questions)
        if len(keys) <= self.batch:
            return self._one(state, questions)
        chunks = [{k: questions[k] for k in keys[i:i + self.batch]}
                  for i in range(0, len(keys), self.batch)]
        label, out = self.label, {}
        def run(chunk):
            self.label = label
            return self._one(state, chunk)
        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            for part in pool.map(run, chunks):
                out.update(part)
        return out

    def ask_many(self, jobs):
        """[(state, questions), ...] -> [answers, ...], run concurrently."""
        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda j: self.ask(*j), jobs))


# ---- question constructors (mirror the three Jev primitives) -------------
def choice(instructions: str, criteria: dict) -> dict:
    if len(criteria) > MAX_OPTIONS:
        criteria = dict(list(criteria.items())[:MAX_OPTIONS])
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": criteria[:10]}


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def pick(answer: dict, default=None):
    """Read whichever field an Answer carries."""
    if not answer:
        return default
    return answer.get("choice", answer.get("score", answer.get("noul", default)))


def probs(answer: dict) -> dict:
    return (answer or {}).get("probabilities", {}) or {}
