"""Two instruments, because forced choice alone cannot see what V4 and V5 do.

CLOZE
    Minimal pairs. Every distractor shares a part of speech with the answer, so
    "pick the only auxiliary" or "pick the only noun" wins nothing. Option counts
    vary, so accuracy is reported against each item's own chance level.

GENERATION
    The network writes a reply and ordinary Python grades it. No Jev in the
    scoring loop, so this is an independent instrument and it measures the part
    of the system forced choice never touches.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import re
import time

from . import db
from .net import ARCHS, Net

# (id, category, context, options, answer, question)
CLOZE = [
    # --- agreement: number on the verb, with nothing in the way ---------------
    ("ag1", "agreement", "The dogs ___ barking .", ["are", "is"], "are", None),
    ("ag2", "agreement", "The cat ___ sleeping .", ["is", "are"], "is", None),
    ("ag3", "agreement", "My friends ___ here now .", ["are", "is"], "are", None),
    ("ag4", "agreement", "The water ___ cold .", ["is", "are"], "is", None),
    ("ag5", "agreement", "Those men ___ tired .", ["are", "is"], "are", None),
    ("ag6", "agreement", "This child ___ happy .", ["is", "are"], "is", None),
    ("ag7", "agreement", "Both girls ___ young .", ["are", "is"], "are", None),
    ("ag8", "agreement", "Every boy ___ hungry .", ["is", "are"], "is", None),
    ("ag9", "agreement", "The books ___ heavy .", ["are", "is"], "are", None),
    ("ag10", "agreement", "The teacher ___ ready .", ["is", "are"], "is", None),

    # --- attraction: a noun of the wrong number sits between subject and verb --
    ("at1", "attraction", "The keys on the table ___ mine .", ["are", "is"], "are", None),
    ("at2", "attraction", "The box of old books ___ heavy .", ["is", "are"], "is", None),
    ("at3", "attraction", "The man who lives in the houses ___ old .", ["is", "are"], "is", None),
    ("at4", "attraction", "The children who saw the film ___ happy .", ["were", "was"], "were", None),
    ("at5", "attraction", "The girl with the two dogs ___ walking .", ["is", "are"], "is", None),
    ("at6", "attraction", "The dogs near the quiet house ___ barking .", ["are", "is"], "are", None),
    ("at7", "attraction", "The colour of the flowers ___ bright .", ["is", "are"], "is", None),
    ("at8", "attraction", "The students in the class ___ waiting .", ["are", "is"], "are", None),
    ("at9", "attraction", "The price of the tickets ___ high .", ["is", "are"], "is", None),
    ("at10", "attraction", "The windows in the big room ___ open .", ["are", "is"], "are", None),

    # --- binding: what the pronoun points at ----------------------------------
    ("bi1", "binding", "The boy hurt ___ .",
     ["himself", "herself", "itself", "themselves"], "himself", None),
    ("bi2", "binding", "The woman bought ___ a hat .",
     ["herself", "himself", "itself", "themselves"], "herself", None),
    ("bi3", "binding", "The dogs found ___ some food .",
     ["themselves", "himself", "herself", "itself"], "themselves", None),
    ("bi4", "binding", "The girl saw ___ in the mirror .",
     ["herself", "himself", "itself", "themselves"], "herself", None),
    ("bi5", "binding", "The cat washed ___ .",
     ["itself", "himself", "herself", "themselves"], "itself", None),
    ("bi6", "binding", "The boy and the girl hurt ___ .",
     ["themselves", "himself", "herself", "itself"], "themselves", None),
    ("bi7", "binding", "John gave Mary a book because ___ had two .",
     ["he", "she", "it", "they"], "he", None),
    ("bi8", "binding", "Mary gave John a book because ___ had two .",
     ["she", "he", "it", "they"], "she", None),
    ("bi9", "binding", "The men helped ___ .",
     ["themselves", "himself", "herself", "itself"], "themselves", None),
    ("bi10", "binding", "The woman said ___ was ready .",
     ["she", "he", "it", "they"], "she", None),

    # --- case: subject or object form of the same pronoun ---------------------
    ("ca1", "case", "Between you and ___ , this is wrong .", ["me", "I"], "me", None),
    ("ca2", "case", "___ and I went home .", ["He", "Him"], "He", None),
    ("ca3", "case", "She gave the book to ___ .", ["me", "I"], "me", None),
    ("ca4", "case", "The teacher saw ___ .", ["us", "we"], "us", None),
    ("ca5", "case", "___ is my friend .", ["He", "Him"], "He", None),
    ("ca6", "case", "Give the book to ___ .", ["him", "he"], "him", None),
    ("ca7", "case", "Anna and ___ are sisters .", ["I", "me"], "I", None),
    ("ca8", "case", "They waited for ___ .", ["him", "he"], "him", None),
    ("ca9", "case", "___ are late again .", ["They", "Them"], "They", None),
    ("ca10", "case", "The dog followed ___ home .", ["them", "they"], "them", None),

    # --- tense: four forms of one verb, only one fits the frame ---------------
    ("te1", "tense", "Yesterday he ___ home .",
     ["walked", "walks", "walking", "walk"], "walked", None),
    ("te2", "tense", "They are ___ in the garden .",
     ["playing", "played", "plays", "play"], "playing", None),
    ("te3", "tense", "Last week we ___ the film .",
     ["saw", "see", "seeing", "sees"], "saw", None),
    ("te4", "tense", "She wants to ___ now .",
     ["leave", "left", "leaves", "leaving"], "leave", None),
    ("te5", "tense", "Every day she ___ to school .",
     ["goes", "went", "going", "gone"], "goes", None),
    ("te6", "tense", "They ___ the house last year .",
     ["built", "build", "builds", "building"], "built", None),
    ("te7", "tense", "I will ___ you tomorrow .",
     ["call", "called", "calls", "calling"], "call", None),
    ("te8", "tense", "He has ___ the letter already .",
     ["written", "writes", "writing", "write"], "written", None),
    ("te9", "tense", "She was ___ when I arrived .",
     ["waiting", "waited", "waits", "wait"], "waiting", None),
    ("te10", "tense", "We did not ___ the answer .",
     ["know", "knew", "knows", "knowing"], "know", None),

    # --- quantifier: count against mass ---------------------------------------
    ("qu1", "quantifier", "There is too ___ water .", ["much", "many"], "much", None),
    ("qu2", "quantifier", "There are too ___ people .", ["many", "much"], "many", None),
    ("qu3", "quantifier", "She has very ___ money .", ["little", "few"], "little", None),
    ("qu4", "quantifier", "He has very ___ friends .", ["few", "little"], "few", None),
    ("qu5", "quantifier", "I drank ___ of the milk .", ["some", "every"], "some", None),
    ("qu6", "quantifier", "She ate ___ apple .", ["an", "a"], "an", None),
    ("qu7", "quantifier", "He bought ___ new car .", ["a", "an"], "a", None),
    ("qu8", "quantifier", "___ of the books are new .", ["Some", "Every"], "Some", None),
    ("qu9", "quantifier", "There is ___ bread left .", ["little", "few"], "little", None),
    ("qu10", "quantifier", "We need ___ chairs than that .", ["more", "most"], "more", None),

    # --- recall: every option appears in the passage, so copying decides nothing
    ("re1", "recall", "John bought milk . Mary bought bread and coffee and water .",
     ["milk", "bread", "coffee", "water"], "milk", "What did John buy?"),
    ("re2", "recall", "The key is in the bag . The book is on the table in the room .",
     ["bag", "table", "room", "book"], "bag", "Where is the key?"),
    ("re3", "recall",
     "Anna has a red car . Tom has a blue car . Sam has a green car . "
     "Kim has a black car .",
     ["red", "blue", "green", "black"], "red", "What colour is Anna's car?"),
    ("re4", "recall", "Sam ate bread and drank water . Lisa ate fish and drank milk .",
     ["water", "bread", "milk", "fish"], "water", "What did Sam drink?"),
    ("re5", "recall",
     "Lisa went to the store . Ben went to the park . Kim went to the school . "
     "Sam went home .",
     ["park", "store", "school", "home"], "park", "Where did Ben go?"),
    ("re6", "recall",
     "The cat is under the table . The dog is behind the door . "
     "The bird is on the roof . The fish is in the water .",
     ["behind", "under", "in", "on"], "behind", "Where is the dog?"),
    ("re7", "recall",
     "Tom gave Mary a book . Mary gave Tom a hat . Ben gave Kim a bag . "
     "Kim gave Ben a letter .",
     ["book", "hat", "bag", "letter"], "book", "What did Tom give?"),
    ("re8", "recall", "We ate at six and left at eight . They ate at seven and left at nine .",
     ["eight", "six", "seven", "nine"], "eight", "When did we leave?"),
    ("re9", "recall",
     "The red box is empty . The blue box is full . The green box is open . "
     "The black box is closed .",
     ["blue", "red", "green", "black"], "blue", "Which box is full?"),
    ("re10", "recall", "The boy is tall . The girl is short . The man is old . The woman is young .",
     ["boy", "girl", "man", "woman"], "boy", "Who is tall?"),

    # --- selection: which verb takes this object ------------------------------
    ("se1", "selection", "She ___ the ball to the dog .",
     ["threw", "slept", "laughed", "arrived"], "threw", None),
    ("se2", "selection", "He ___ the cold water .",
     ["drank", "walked", "sang", "waited"], "drank", None),
    ("se3", "selection", "The bird ___ away from the tree .",
     ["flew", "ate", "wrote", "bought"], "flew", None),
    ("se4", "selection", "She ___ the heavy door .",
     ["opened", "drank", "slept", "flew"], "opened", None),
    ("se5", "selection", "He ___ a long letter .",
     ["wrote", "ran", "slept", "flew"], "wrote", None),
    ("se6", "selection", "They ___ the old song together .",
     ["sang", "drank", "walked", "slept"], "sang", None),
    ("se7", "selection", "The dog ___ the bone .",
     ["ate", "wrote", "drove", "sang"], "ate", None),
    ("se8", "selection", "She ___ the car to work .",
     ["drove", "ate", "sang", "slept"], "drove", None),
    ("se9", "selection", "The man ___ the money carefully .",
     ["counted", "slept", "flew", "sang"], "counted", None),
    ("se10", "selection", "He ___ the music on the radio .",
     ["heard", "drank", "drove", "built"], "heard", None),

    # --- plausibility: which one does the world allow -------------------------
    ("pl1", "plausibility", "The ice sat in the hot sun and ___ .",
     ["melted", "grew", "sang", "waited"], "melted", None),
    ("pl2", "plausibility", "The glass fell on the stone floor and ___ .",
     ["broke", "grew", "sang", "slept"], "broke", None),
    ("pl3", "plausibility", "It rained all morning , so the street was ___ .",
     ["wet", "dry", "hot", "loud"], "wet", None),
    ("pl4", "plausibility", "The fire made the cold room ___ .",
     ["warm", "cold", "wet", "quiet"], "warm", None),
    ("pl5", "plausibility", "She was very tired , so she went to ___ .",
     ["sleep", "work", "run", "sing"], "sleep", None),
    ("pl6", "plausibility", "The sun came up and the day became ___ .",
     ["bright", "dark", "wet", "loud"], "bright", None),
    ("pl7", "plausibility", "Winter is ___ than summer .",
     ["colder", "hotter", "louder", "wetter"], "colder", None),
    ("pl8", "plausibility", "He ate the bread because he was ___ .",
     ["hungry", "tired", "happy", "tall"], "hungry", None),
    ("pl9", "plausibility", "The box was very heavy and hard to ___ .",
     ["lift", "read", "hear", "taste"], "lift", None),
    ("pl10", "plausibility", "He had no money at all , so he ___ not buy it .",
     ["could", "should", "would", "will"], "could", None),
]

CATEGORIES = ["agreement", "attraction", "binding", "case", "tense",
              "quantifier", "recall", "selection", "plausibility"]

# Generation probes. Graded by the functions below, not by Jev.
PROMPTS = [
    "hello",
    "what is a cat ?",
    "what colour is the sky ?",
    "where is the dog ?",
    "is the water cold ?",
    "tell me about the old house",
    "the dogs near the river",
    "why was the boy sad ?",
]

VOWEL = "aeiou"
STOP = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "and",
        "or", "but", "it", "that", "this", "what", "where", "why", "who", "me", "my"}


# ---- deterministic graders ------------------------------------------------
def _bigrams(words):
    return list(zip(words, words[1:]))


def grade(words: list[str], prompt: str, vocab, hit_cap: bool) -> dict[str, int]:
    """Ordinary Python marking a reply. Every check is 1 for good, 0 for bad."""
    low = [w.lower() for w in words]
    content = [w for w in low if w not in STOP and w.isalpha()]
    tags = [vocab.tag_of(w) or vocab.tag_of(w.lower()) or "?" for w in words]
    prompt_words = {w.lower().strip("?.,!") for w in prompt.split()}

    agreement = 1
    for (a, b), (ta, _) in zip(_bigrams(low), _bigrams(tags)):
        if ta == "NPL" and b in ("is", "was"):
            agreement = 0
        if ta == "N" and b in ("are", "were"):
            agreement = 0

    determiner = 1
    for a, b in _bigrams(low):
        if b and b[0].isalpha():
            if a == "a" and b[0] in VOWEL:
                determiner = 0
            if a == "an" and b[0] not in VOWEL:
                determiner = 0

    return {
        "produced": int(bool(words)),
        "no_stutter": int(all(a != b for a, b in _bigrams(low))),
        "varied": int(not low or len(set(low)) / len(low) >= 0.7),
        "agreement": agreement,
        "determiner": determiner,
        "has_verb": int(any(t in ("V", "VS", "VD", "VG", "AUXS", "AUXP", "MODAL")
                            for t in tags)),
        "not_echo": int(bool(content) and not set(content) <= prompt_words),
        "stopped": int(not hit_cap),
    }


CHECKS = ["produced", "no_stutter", "varied", "agreement", "determiner",
          "has_verb", "not_echo", "stopped"]


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval. With 10 items per category, this is the honest error bar."""
    if not n:
        return (0.0, 0.0)
    p, d = hits / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def select(category: str | None = None, limit: int = 0):
    """limit is per category, so a small run still spans every category."""
    rows = [t for t in CLOZE if not category or t[1] == category]
    if not limit:
        return rows
    seen: dict[str, int] = {}
    out = []
    for task in rows:
        seen[task[1]] = seen.get(task[1], 0) + 1
        if seen[task[1]] <= limit:
            out.append(task)
    return out


def audit() -> list[str]:
    """The benchmark checks itself: no item may be solvable by a cheap trick.

    The old suite failed this - 23 of its 40 items could be answered by picking
    the only option of the right part of speech.
    """
    from . import vocab as vocab_mod
    v = vocab_mod.build(10 ** 6)
    problems = []
    for tid, cat, ctx, opts, ans, q in CLOZE:
        if ans not in opts:
            problems.append(f"{tid}: answer not among the options")
        if len(opts) != len(set(opts)):
            problems.append(f"{tid}: duplicate options")
        roles = set()
        unknown = 0
        for opt in opts:
            tag = v.tag_of(opt) or v.tag_of(opt.lower())
            if tag is None:
                unknown += 1
                continue
            roles |= {r for r, tags in vocab_mod.ROLES.items() if tag in tags} or {tag}
        if len(roles) > 1:
            problems.append(f"{tid}: options fill different slots {sorted(roles)} - "
                            f"the syntactic position alone narrows it")
        if unknown == len(opts):
            problems.append(f"{tid}: no option is in the lexicon - cannot verify")
        if cat == "recall":
            body = {w.lower().strip(".,?") for w in ctx.split()}
            missing = [o for o in opts if o.lower() not in body]
            if missing:
                problems.append(f"{tid}: {missing} absent from the passage - "
                                f"copying would find the answer")
    return problems


# ---- running ---------------------------------------------------------------
def run(con, arch_name: str, jev, vocab, tasks, prompts=(), workers: int = 4,
        verbose: bool = True, on_result=None, gen_words: int = 8) -> dict:
    arch = ARCHS[arch_name]
    run_id = db.start_run(con, "bench", arch_name, f"{len(tasks)}+{len(prompts)}")
    started = time.time()

    def one(task):
        tid, category, context, options, answer, question = task
        net = Net(arch, jev, vocab, temperature=0.0)
        dist = net.score_options(context, options, question)
        chosen = max(dist, key=dist.get) if dist else ""
        return tid, category, chosen, answer, dist, len(options)

    results = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for done in cf.as_completed([pool.submit(one, t) for t in tasks]):
            results.append(done.result())
            if on_result:
                tid, cat, chosen, answer, dist, _ = results[-1]
                on_result(tid, cat, chosen, answer, dist)

    tally: dict[str, list[int]] = {}
    chance_sum = hits = 0
    for tid, category, chosen, answer, dist, n_opts in results:
        db.add_eval(con, run_id, arch_name, tid, category, chosen, answer, dist)
        hit = int(chosen == answer)
        hits += hit
        chance_sum += 1 / n_opts
        tally.setdefault(category, []).append(hit)
        if verbose:
            print(f"  {'ok ' if hit else 'MISS'} {tid:<5}{category:<14}"
                  f"chose {chosen!r:<14} want {answer!r}")

    n = len(results)
    by_cat = {c: {"accuracy": sum(v) / len(v), "n": len(v),
                  "ci": wilson(sum(v), len(v))} for c, v in tally.items()}
    accuracy = hits / n if n else 0.0
    chance = chance_sum / n if n else 0.0

    # --- generation ---------------------------------------------------------
    gen_rows, gen = [], {}
    if prompts:
        def write(prompt):
            net = Net(arch, jev, vocab, temperature=0.7)
            words = net.generate(prompt, gen_words, reply=True)
            marks = grade(words, prompt, vocab, hit_cap=len(words) >= gen_words)
            return prompt, words, marks

        with cf.ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
            gen_rows = list(pool.map(write, prompts))
        gen = {k: sum(m[k] for _, _, m in gen_rows) / len(gen_rows) for k in CHECKS}
        gen["overall"] = sum(gen[k] for k in CHECKS) / len(CHECKS)
        if verbose:
            for prompt, words, marks in gen_rows:
                failed = [k for k in CHECKS if not marks[k]]
                print(f"  {prompt!r} -> {' '.join(words)!r}"
                      + (f"   fails: {', '.join(failed)}" if failed else "   clean"))

    meta = {"jevs": arch.jevs, "accuracy": accuracy, "chance": chance,
            "lift": accuracy - chance, "by_category": by_cat, "tasks": n,
            "generation": gen,
            "samples": [{"prompt": p, "reply": " ".join(w), "marks": m}
                        for p, w, m in gen_rows]}
    db.finish_run(con, run_id, f"accuracy={accuracy:.3f} lift={accuracy - chance:+.3f}",
                  jev.usage, time.time() - started, meta)
    return {"run_id": run_id, "accuracy": accuracy, "chance": chance,
            "lift": accuracy - chance, "ci": wilson(hits, n),
            "by_category": by_cat, "generation": gen, "samples": meta["samples"],
            "usage": dict(jev.usage), "seconds": time.time() - started}


def table(rows: dict) -> str:
    """rows: {arch_name: result} -> a comparison, with error bars."""
    head = (f"{'arch':<6}{'jevs':>6}{'acc':>7}{'chance':>8}{'lift':>8}"
            f"{'95% CI':>16}{'gen':>7}")
    out = [head, "-" * len(head)]
    for name, r in rows.items():
        lo, hi = r["ci"]
        g = r["generation"].get("overall")
        out.append(f"{name:<6}{ARCHS[name].jevs:>6}{r['accuracy']:>7.2f}"
                   f"{r['chance']:>8.2f}{r['lift']:>+8.2f}"
                   f"{f'[{lo:.2f}, {hi:.2f}]':>16}"
                   f"{(f'{g:.2f}' if g is not None else '-'):>7}")
    cats = sorted({c for r in rows.values() for c in r["by_category"]})
    if cats:
        out.append("")
        out.append(f"{'category':<14}" + "".join(f"{n:>8}" for n in rows))
        for c in cats:
            cells = "".join(
                f"{r['by_category'][c]['accuracy']:>8.2f}" if c in r["by_category"]
                else f"{'-':>8}" for r in rows.values())
            out.append(f"{c:<14}{cells}")
    return "\n".join(out)
