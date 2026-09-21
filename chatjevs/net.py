"""ChatJEVs: a next-word predictor built only out of bounded Jev decisions.

One generation step is a small number of batched requests:

    routing (V3 only) -> perception+syntax+memory+concepts -> decoder levels

Nothing is learned inside Jev. What varies between V0 and V3 is the
architecture around it: how many units run, what they are asked, what state
they see, and how their answers are routed into the decoder.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import banks, vocab as vocab_mod
from .client import choice, noul, pick, probs, score


@dataclass
class Arch:
    name: str
    semantic: int
    syntax: int
    memory: int
    concepts: int
    routing: int
    plan: int            # speech-act units, run once per reply
    critics: int         # candidate x dimension judges before committing a word
    drafts: int          # whole replies written, judged and repaired
    levels: int          # decoder depth: 1 flat, 2 tag->word, 3 tag->bucket->word
    rerank: bool
    recurrent: bool      # feed last step's analysis back in as state
    vocab_size: int
    window: int
    note: str

    draft_with: str = ""     # architecture used to write the drafts
    revise: bool = False

    @property
    def decoder(self) -> int:
        return self.levels + 1 + (1 if self.rerank else 0)   # +1 end-of-text unit

    @property
    def word_jevs(self) -> int:
        """Units consulted to place one word."""
        return (self.semantic + self.syntax + self.memory + self.concepts
                + self.routing + self.decoder + self.critics)

    @property
    def jevs(self) -> int:
        total = self.word_jevs + self.plan
        if self.drafts > 1:
            total += self.drafts * ARCHS[self.draft_with].word_jevs
            total += self.drafts * len(banks.DRAFT_DIMS) + 1      # the judging panel
        if self.revise:
            total += REVISE_SPAN                                  # keep / replace / cut
        return total

    @property
    def requests_per_token(self) -> str:
        lo = 2 + (1 if self.routing else 0)                  # banks + word choice
        hi = lo + (self.levels - 1) + (1 if self.rerank else 0) + (1 if self.critics else 0)
        return f"{lo}-{hi}"


ARCHS = {
    "v0": Arch("v0", 4, 4, 0, 0, 0, 0, 0, 0, 1, False, False, 255, 8,
               "feed-forward, no memory, no plan, flat decoder"),
    "v1": Arch("v1", 8, 8, 8, 2, 0, 2, 0, 0, 2, True, True, 255, 16,
               "semantic + syntax + memory, planned, recurrent state"),
    "v2": Arch("v2", 16, 16, 16, 46, 0, 2, 0, 0, 2, True, True, 5000, 32,
               "concept bank + hierarchical decoder"),
    "v3": Arch("v3", 24, 24, 24, 158, 16, 2, 0, 0, 3, True, True, 20000, 64,
               "dynamic routing, gated banks, deep decoder"),
    "v4": Arch("v4", 24, 24, 24, 240, 16, 2, 48, 0, 3, False, True, 20000, 64,
               "every candidate word is judged before it is committed"),
    "v5": Arch("v5", 24, 24, 24, 240, 16, 2, 48, 3, 3, False, True, 20000, 96,
               "three drafts, a judging panel, then a repair pass",
               draft_with="v1", revise=True),
}

REVISE_SPAN = 24          # positions a revision pass will consider at once
MAX_REPLACEMENTS = 3      # re-decoding is the expensive half; cap it

TASK = ("Predict how this English text continues. Answer each question about the "
        "text as it stands, not about the whole imagined sentence.")

RULE = ("Choose what a fluent native English writer would actually write next. "
        "Stay on the topic of `user_message`. Prefer grammar over novelty, and do "
        "not invent facts that are not in `user_message` unless they are common "
        "knowledge.")

REPLY_TASK = (
    "You are writing a reply to the user's message, one word at a time. "
    "`user_message` is what the user said. `text` is the reply written so far. "
    "Every question below is about that reply: what it needs next in order to be a "
    "grammatical and relevant answer to the user.")

END = "<end>"


def sample(dist: dict, temperature: float = 0.8, top_p: float = 0.95) -> str:
    """Sample a key from a probability map. temperature<=0 is greedy."""
    if not dist:
        return ""
    if temperature <= 0:
        return max(dist, key=dist.get)
    scaled = {k: math.exp(math.log(max(v, 1e-9)) / temperature) for k, v in dist.items()}
    total = sum(scaled.values()) or 1.0
    ranked = sorted(((v / total, k) for k, v in scaled.items()), reverse=True)
    kept, acc = [], 0.0
    for p, k in ranked:
        kept.append((p, k))
        acc += p
        if acc >= top_p:
            break
    r = random.random() * sum(p for p, _ in kept)
    for p, k in kept:
        r -= p
        if r <= 0:
            return k
    return kept[-1][1]


def _salient(answers: dict, limit: int) -> dict:
    """Compact an analysis for feedback: strongest units only, values only."""
    scored = []
    for key, ans in answers.items():
        value = pick(ans)
        if value is None:
            continue
        conf = ans.get("confidence", abs((ans.get("noul") or 0.5) - 0.5) * 2)
        scored.append((conf, key, value))
    scored.sort(reverse=True)
    return {k: v for _, k, v in scored[:limit]}


class Net:
    def __init__(self, arch: Arch, jev, vocab: vocab_mod.Vocab,
                 temperature: float = 0.8, top_p: float = 0.95, seed: int | None = None):
        self.arch, self.jev, self.vocab = arch, jev, vocab
        self.temperature, self.top_p = temperature, top_p
        if seed is not None:
            random.seed(seed)
        self.prompt: str | None = None     # set while replying; None = continuation
        self.plan: dict | None = None
        self.memory: dict[str, str] = {}
        self.analysis: dict = {}
        self.trace: list[dict] = []
        self.on_note = None                # (kind, data) -> None, for the UI

    # ---- state ----------------------------------------------------------
    def _state(self, context: list[str], extra: dict | None = None) -> dict:
        arch, window = self.arch, self.arch.window
        if extra and extra.get("attend_window") == "last_3":
            window = 3
        elif extra and extra.get("attend_window") == "last_8":
            window = 8
        replying = self.prompt is not None
        state = {"task": REPLY_TASK if replying else TASK,
                 "text": " ".join(context) if context else
                         ("(the reply has not started)" if replying else "(nothing written yet)"),
                 "recent_words": context[-window:]}
        if replying:
            state["user_message"] = self.prompt
            state["rule"] = RULE
            if self.plan:
                state["plan"] = self.plan
        if arch.memory and self.memory:
            state["working_memory"] = self.memory
        if arch.recurrent and self.analysis:
            state["previous_analysis"] = self.analysis
        if extra:
            state["routing"] = extra
        return state

    def _make_plan(self, prompt: str) -> dict | None:
        if not self.arch.plan:
            return None
        self.jev.label = "plan"
        answers = self.jev.ask(
            {"task": "Plan a short English reply to `user_message`.",
             "user_message": prompt},
            {"intent": choice(
                "What should the English reply to `user_message` do? "
                "Pick the single best speech act.", {
                    "answer": "Answer the question with a short factual English sentence",
                    "define": "Give a short English definition or explanation",
                    "continue": "Continue the text in `user_message` in the same style",
                    "greet": "Return a short greeting or pleasantry",
                    "clarify": "Ask one short clarifying question in English",
                    "refuse": "Say politely that this cannot be answered"}),
             "length": score("How long should that English reply be?", [
                 "a single word or a short phrase", "one simple sentence",
                 "two short sentences"])})
        length = answers.get("length", {}).get("score", 1.0)
        return {"intent": pick(answers.get("intent")) or "answer",
                "length": "a short phrase" if length < 0.5 else
                          "one sentence" if length < 1.5 else "two sentences",
                "max_words": 4 if length < 0.5 else 12 if length < 1.5 else 20}

    # ---- banks ----------------------------------------------------------
    def _bank_questions(self, gates: dict) -> dict:
        arch = self.arch
        q = banks.take(banks.SEMANTIC, arch.semantic if gates.get("semantics", True) else 2)
        q.update(banks.take(banks.SYNTAX, arch.syntax))          # syntax always runs
        if arch.memory and gates.get("memory", True):
            q.update(banks.take(banks.MEMORY, arch.memory))
        if arch.concepts and gates.get("concepts", True):
            q.update(banks.concept_questions(arch.concepts))
        q["end_of_text"] = noul(
            "The reply is complete: it answers the user's message and no further word "
            "should be added." if self.prompt is not None else
            "This passage is finished: no further word, phrase or sentence should be "
            "added to it, and continuing would produce nonsense.")
        return q

    def _route(self, context: list[str]) -> tuple[dict, dict]:
        if not self.arch.routing:
            return {}, {}
        self.jev.label = "routing"
        answers = self.jev.ask(self._state(context), banks.take(banks.ROUTING, self.arch.routing))
        decisions = {k: pick(v) for k, v in answers.items()}
        gates = {"semantics": decisions.get("run_semantics", 1) >= 0.4,
                 "memory": decisions.get("run_memory", 1) >= 0.4,
                 "concepts": decisions.get("run_concepts", 1) >= 0.5}
        return decisions, gates

    def _remember(self, answers: dict, last_word: str) -> None:
        if not self.arch.memory or not last_word:
            return
        for slot in banks.SLOTS:
            if pick(answers.get(f"{slot.lower()}_status")) == "CLEAR":
                self.memory.pop(slot, None)
        target = pick(answers.get("slot_of_last"))
        if target and target in banks.SLOTS:
            if pick(answers.get(f"{target.lower()}_status", {})) != "KEEP" or target not in self.memory:
                self.memory[target] = last_word
        if (answers.get("sentence_boundary", {}).get("noul") or 0) > 0.7:
            for slot in ("ACTION", "MODIFIER", "RECIPIENT"):
                self.memory.pop(slot, None)

    # ---- decoder --------------------------------------------------------
    def _candidates(self, answers: dict):
        role = pick(answers.get("next_role")) or "NOUN"
        form = pick(answers.get("verb_form"))
        number = pick(answers.get("subject_number"))
        cands = self.vocab.for_role(role)
        if role == "VERB" and form in ("third_singular", "past", "progressive", "base"):
            tag = {"third_singular": "VS", "past": "VD",
                   "progressive": "VG", "base": "V"}[form]
            cands = [c for c in cands if c[1] == tag] or cands
        elif role == "AUXILIARY" and number in ("singular", "plural"):
            keep = "AUXS" if number == "singular" else "AUXP"
            preferred = [c for c in cands if c[1] in (keep, "MODAL", "NEG", "PART")]
            cands = preferred or cands
        return cands, role

    def _copied(self, context: list[str]) -> list[tuple[str, str]]:
        """Words from the conversation are always candidates. With a small
        vocabulary this is most of what keeps a reply on topic."""
        out, seen = [], set()
        for raw in (str(self.prompt or "") + " " + " ".join(context)).split():
            word = raw.strip('.,!?;:"\'()[]')
            if not word or word.lower() in seen:
                continue
            seen.add(word.lower())
            tag = self.vocab.tag_of(word) or self.vocab.tag_of(word.lower()) or "N"
            out.append((word, tag))
        return out[:24]

    def _pool(self, preferred, context: list[str], limit: int):
        """Preferred words first, then conversation words, then frequent ones,
        so a wrong guess by the syntax units is still recoverable."""
        seen, out = set(), []
        # the last group only fills budget the first two left unused
        for group in (preferred, self._copied(context), self.vocab.words):
            for word, tag in group:
                if word in seen:
                    continue
                seen.add(word)
                out.append((word, tag))
                if len(out) >= limit:
                    return out
        return out

    def _decode(self, state: dict, answers: dict, context: list[str],
                allow_end: bool = False):
        cands, role = self._candidates(answers)
        levels, path = self.arch.levels, {}
        if levels > 1 and len({t for _, t in cands}) > 1:
            tags = [t for t in self.vocab.tags if any(t == ct for _, ct in cands)]
            self.jev.label = "form"
            answer = self.jev.ask(state, {"form": choice(
                f"Which form of {role.lower()} should come next?",
                {t: vocab_mod.GLOSS[t] for t in tags})})["form"]
            tag = sample(probs(answer), self.temperature, 1.0) or pick(answer)
            path["form"] = (tag, probs(answer))
            cands = [c for c in cands if c[1] == tag] or cands
        if (levels > 2 and len(cands) > 64) or len(cands) > 255:
            groups = vocab_mod.buckets(cands)
            if len(groups) > 1:
                self.jev.label = "bucket"
                answer = self.jev.ask(state, {"bucket": choice(
                    "Which group holds the next English word?",
                    vocab_mod.bucket_criteria(groups[:255]))})["bucket"]
                label = sample(probs(answer), self.temperature, 1.0) or pick(answer)
                path["bucket"] = (label, probs(answer))
                cands = dict(groups).get(label, cands[:255])
        self.jev.label = "word"
        cands = self._pool(cands, context, 254 if allow_end else 255)
        criteria = self.vocab.criteria(cands, limit=254 if allow_end else 255)
        if allow_end:
            criteria[END] = "the reply is finished; write nothing more"
        answer = self.jev.ask(state, {"word": choice(
            "Which single word comes next in the reply to the user?" if self.prompt
            else "Which single word comes next in the text?", criteria)})["word"]
        dist = probs(answer)
        path["word"] = (pick(answer), dist)
        word = sample(dist, self.temperature, self.top_p) or pick(answer) or ""
        if self.arch.critics and len(dist) > 1:
            dist = self._critique(state, dist)
            word = sample(dist, self.temperature, self.top_p) or word
            path["critics"] = (word, dist)
        elif self.arch.rerank and len(dist) > 1:
            top = dict(sorted(dist.items(), key=lambda kv: -kv[1])[:8])
            if word not in top:
                top[word] = dist.get(word, 0.0)
            self.jev.label = "rerank"
            answer = self.jev.ask(state, {"rerank": choice(
                "Of these shortlisted words, which truly reads best as the next word?",
                {w: "candidate" for w in top})})["rerank"]
            dist = probs(answer) or dist
            word = sample(dist, self.temperature, self.top_p) or word
            path["rerank"] = (word, dist)
        return word, dist

    # ---- judgment (V4) --------------------------------------------------
    def _critique(self, state: dict, dist: dict) -> dict:
        """Score the top candidates with a panel of critics and re-weight.

        Asking "is this word right here?" plays to what a decision model does
        well, where "pick one of 255" does not.
        """
        width = max(2, self.arch.critics // len(banks.CRITIC_DIMS))
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        top = [w for w, _ in ranked[:width] if w != END]
        if len(top) < 2:
            return dist
        self.jev.label = "critics"
        scores = banks.critic_scores(
            top, self.jev.ask(state, banks.critic_questions(top)))
        # geometric blend: the decoder proposes, the critics dispose
        blended = {w: (dist[w] ** 0.4) * (max(scores[w], 1e-3) ** 0.6) for w in top}
        if END in dist:
            blended[END] = dist[END]
        total = sum(blended.values()) or 1.0
        return {w: v / total for w, v in blended.items()}

    # ---- deliberation (V5) ----------------------------------------------
    def _note(self, kind: str, data) -> None:
        if self.on_note:
            self.on_note(kind, data)

    def _judge(self, prompt: str, drafts: list[list[str]]) -> int:
        texts = {f"draft_{i + 1}": " ".join(d) or "(empty)" for i, d in enumerate(drafts)}
        self.jev.label = "judge"
        questions = banks.draft_questions(len(drafts))
        questions["best"] = choice(
            "Which draft is the best reply to the user's message?",
            {key: f"the reply {text!r}" for key, text in texts.items()})
        answers = self.jev.ask(
            {"task": "Judge these candidate replies to `user_message`.",
             "user_message": prompt, "drafts": texts, "rule": RULE}, questions)
        panel = banks.draft_scores(len(drafts), answers)
        vote = probs(answers.get("best"))
        merged = [panel[i] + vote.get(f"draft_{i + 1}", 0.0) for i in range(len(drafts))]
        self._note("judge", {"scores": [round(m, 3) for m in merged],
                             "drafts": [" ".join(d) for d in drafts]})
        return max(range(len(drafts)), key=lambda i: merged[i])

    def _revise(self, prompt: str, words: list[str]) -> list[str]:
        if not words:
            return words
        span = words[:REVISE_SPAN]
        self.jev.label = "revise"
        answers = self.jev.ask(
            {"task": "Repair this draft reply to `user_message`, word by word.",
             "user_message": prompt, "draft": " ".join(span), "rule": RULE},
            banks.revise_questions(span))
        verdicts = [pick(answers.get(f"w{i}")) or "KEEP" for i in range(len(span))]
        self._note("revise", {"words": span, "verdicts": verdicts})

        out, replaced = [], 0
        for word, verdict in zip(span, verdicts):
            if verdict == "CUT":
                continue
            if verdict == "REPLACE" and replaced < MAX_REPLACEMENTS:
                replaced += 1
                fresh, dist, _ = self.step([w for w, _ in out], allow_stop=False)
                out.append((fresh or word, dist))
                continue
            out.append((word, {word: 1.0}))
        return out + [(w, {w: 1.0}) for w in words[REVISE_SPAN:]]

    def _deliberate(self, prompt: str, max_tokens: int, on_token) -> list[str]:
        """Write it several times fast, judge, then repair the winner."""
        drafts = []
        for i in range(self.arch.drafts):
            self.jev.label = f"draft {i + 1}"
            scribe = Net(ARCHS[self.arch.draft_with], self.jev, self.vocab,
                         temperature=0.35 + 0.3 * i, top_p=self.top_p)
            words = scribe.generate(prompt, max_tokens, reply=True)
            drafts.append(words)
            self._note("draft", {"index": i + 1, "text": " ".join(words)})
        self.prompt, self.plan = prompt, self.plan or self._make_plan(prompt)
        winner = drafts[self._judge(prompt, drafts)]
        final = (self._revise(prompt, winner) if self.arch.revise
                 else [(w, {w: 1.0}) for w in winner])
        for word, dist in final:
            self.trace.append({"token": word, "memory": dict(self.memory),
                               "analysis": dict(self.analysis)})
            if on_token:
                on_token(word, dist, dict(self.memory))
        return [w for w, _ in final]

    # ---- generation -----------------------------------------------------
    def step(self, context: list[str], allow_stop: bool = True) -> tuple[str, dict, dict]:
        decisions, gates = self._route(context)
        state = self._state(context, decisions or None)
        self.jev.label = "banks"
        answers = self.jev.ask(state, self._bank_questions(gates))
        self._remember(answers, context[-1] if context else "")
        if self.arch.recurrent:
            self.analysis = _salient(answers, 40)
        role_answer = answers.get("next_role")
        stopping = ((answers.get("end_of_text", {}).get("noul") or 0) > 0.8
                    or pick(role_answer) == "END")
        if stopping and allow_stop:
            return "", {}, answers
        if pick(role_answer) == "END":
            # forced to continue: fall back to the best role that is not END
            rest = {k: v for k, v in probs(role_answer).items() if k != "END"}
            if rest:
                answers["next_role"] = dict(role_answer, choice=max(rest, key=rest.get))
        if decisions.get("commit", 0) > 0.8:
            self.temperature, saved = 0.0, self.temperature
        else:
            saved = self.temperature
        word, dist = self._decode(self._state(context, decisions or None), answers,
                                  context,
                                  allow_end=allow_stop and self.prompt is not None)
        self.temperature = saved
        return word, dist, answers

    def generate(self, prompt: str, max_tokens: int = 12, on_token=None,
                 reply: bool = True) -> list[str]:
        """reply=True answers the prompt; reply=False continues it."""
        self.prompt = prompt if reply else None
        self.plan = self._make_plan(prompt) if reply else None
        if reply and self.arch.drafts > 1:
            return self._deliberate(prompt, max_tokens, on_token)
        if self.plan:
            max_tokens = min(max_tokens, self.plan["max_words"])
        context = [] if reply else prompt.split()
        produced = []
        for _ in range(max_tokens):
            # never return nothing: the first word is always produced
            word, dist, answers = self.step(context, allow_stop=bool(produced))
            if not word or word == END:
                break
            context.append(word)
            produced.append(word)
            self.trace.append({"token": word, "memory": dict(self.memory),
                               "analysis": dict(self.analysis)})
            if on_token:
                on_token(word, dist, dict(self.memory))
            if not reply and word in (".", "!", "?"):
                break
        return produced

    # ---- evaluation -----------------------------------------------------
    def score_options(self, context: str, options: list[str], question: str | None = None) -> dict:
        """Forced choice: run the full network, then restrict the decoder to `options`."""
        self.prompt, self.plan = None, None   # the benchmark is fill-in-the-blank
        words = context.split()
        _, gates = self._route(words)
        state = self._state(words)
        self.jev.label = "banks"
        answers = self.jev.ask(state, self._bank_questions(gates))
        self._remember(answers, words[-1] if words else "")
        if self.arch.recurrent:
            self.analysis = _salient(answers, 40)
        state = self._state(words)
        instructions = question or "Which word correctly fills the blank marked ___ ?"
        self.jev.label = "answer"
        answer = self.jev.ask(state, {"fill": choice(
            instructions,
            {o: f"the answer is '{o}'" for o in options})})["fill"]
        dist = probs(answer) or {pick(answer): 1.0}
        return self._critique(state, dist) if self.arch.critics else dist
