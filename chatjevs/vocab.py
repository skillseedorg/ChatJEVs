"""Vocabulary: a tagged word list, built by inflecting a small lexicon of stems.

The lexicon ships ~375 stems; inflection expands it to ~670 words. Anything
bigger is imported (``chatjevs vocab import file.txt``) and stored in the db.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

LEXICON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lexicon.txt")

GLOSS = {
    "DET": "determiner", "PRON": "pronoun (subject)", "PRONO": "pronoun (object)",
    "AUXS": "auxiliary, singular subject", "AUXP": "auxiliary, plural subject",
    "MODAL": "modal verb", "NEG": "negation", "PART": "particle",
    "N": "noun, singular", "NPL": "noun, plural", "V": "verb, base or plural present",
    "VS": "verb, 3rd person singular", "VD": "verb, past tense", "VG": "verb, -ing form",
    "ADJ": "adjective", "ADJC": "adjective, comparative", "ADV": "adverb", "PREP": "preposition", "CONJ": "conjunction",
    "WH": "question word", "NUM": "number", "PUNCT": "punctuation",
}

# Grammatical role (what a syntax Jev picks) -> which tags may fill it.
ROLES = {
    "DETERMINER": ("DET", "NUM"),
    "NOUN": ("N", "NPL"),
    "PRONOUN": ("PRON", "PRONO"),
    "VERB": ("V", "VS", "VD", "VG"),
    "AUXILIARY": ("AUXS", "AUXP", "MODAL", "NEG", "PART"),
    "ADJECTIVE": ("ADJ", "ADJC"),
    "ADVERB": ("ADV",),
    "PREPOSITION": ("PREP",),
    "CONJUNCTION": ("CONJ",),
    "QUESTION_WORD": ("WH",),
    "PUNCTUATION": ("PUNCT",),
}

IRREG_PLURAL = {"man": "men", "woman": "women", "child": "children", "person": "people",
                "foot": "feet", "mouse": "mice", "fish": "fish", "sheep": "sheep"}
IRREG_THIRD = {"have": "has", "do": "does", "go": "goes", "be": "is"}
IRREG_PAST = {"eat": "ate", "run": "ran", "sleep": "slept", "see": "saw", "take": "took",
              "give": "gave", "make": "made", "go": "went", "come": "came", "know": "knew",
              "think": "thought", "say": "said", "find": "found", "sit": "sat", "stand": "stood",
              "read": "read", "write": "wrote", "hold": "held", "buy": "bought", "sell": "sold",
              "drink": "drank", "hear": "heard", "feel": "felt", "bring": "brought",
              "tell": "told", "keep": "kept", "put": "put", "break": "broke", "build": "built",
              "catch": "caught", "send": "sent", "lose": "lost", "win": "won", "leave": "left",
              "cut": "cut", "hit": "hit", "let": "let", "set": "set", "shut": "shut",
              "cost": "cost", "hurt": "hurt", "quit": "quit", "become": "became",
              "begin": "began", "drive": "drove", "fall": "fell", "feel": "felt",
              "fly": "flew", "forget": "forgot", "grow": "grew", "hear": "heard",
              "hold": "held", "meet": "met", "pay": "paid", "rise": "rose",
              "sing": "sang", "sit": "sat", "speak": "spoke", "spend": "spent",
              "teach": "taught", "tell": "told", "throw": "threw", "wear": "wore"}


def _plural(w: str) -> str:
    if w in IRREG_PLURAL:
        return IRREG_PLURAL[w]
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def _third(w: str) -> str:
    if w in IRREG_THIRD:
        return IRREG_THIRD[w]
    if w.endswith("o"):                      # go -> goes, echo -> echoes
        return w + "es"
    return _plural(w) if w not in IRREG_PLURAL else w + "s"


def _past(w: str) -> str:
    if w in IRREG_PAST:
        return IRREG_PAST[w]
    if w.endswith("e"):
        return w + "d"
    if w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ied"
    return w + "ed"


IRREG_COMPARATIVE = {"good": "better", "bad": "worse", "far": "further",
                     "many": "more", "much": "more", "little": "less"}


def _comparative(w: str) -> str | None:
    """-er on the short adjectives only; 'more careful' is two words."""
    if w in IRREG_COMPARATIVE:
        return IRREG_COMPARATIVE[w]
    if len(w) > 7 or w.endswith(("er", "ed", "ing", "ly", "ous", "ful", "ent")):
        return None
    if w.endswith("e"):
        return w + "r"
    if w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ier"
    if (len(w) > 2 and w[-1] not in "aeiouwy"
            and w[-2] in "aeiou" and w[-3] not in "aeiou"):
        return w + w[-1] + "er"          # big -> bigger
    return w + "er"


def _ing(w: str) -> str:
    if w.endswith("e") and len(w) > 2:
        return w[:-1] + "ing"
    return w + "ing"


@dataclass
class Vocab:
    words: list[tuple[str, str]]          # (word, tag), ranked

    def __len__(self) -> int:
        return len(self.words)

    @property
    def tags(self) -> list[str]:
        seen = []
        for _, t in self.words:
            if t not in seen:
                seen.append(t)
        return seen

    def by_tags(self, tags) -> list[tuple[str, str]]:
        tags = set(tags)
        return [wt for wt in self.words if wt[1] in tags]

    def for_role(self, role: str) -> list[tuple[str, str]]:
        hits = self.by_tags(ROLES.get(role, ()))
        return hits or self.words

    def criteria(self, words, limit: int = 255) -> dict[str, str]:
        """Choice criteria, capped at Jev's 255 options.

        The description has to name the word. A bare part-of-speech gloss makes
        every noun in the list read identically and the decision impossible.
        """
        out: dict[str, str] = {}
        for w, t in words:
            if w not in out:
                out[w] = f"The next English word is '{w}' ({GLOSS.get(t, t)})"
            if len(out) >= limit:
                break
        return out

    def tag_of(self, word: str) -> str | None:
        if not hasattr(self, "_index"):
            self._index = {w: t for w, t in self.words}
        return self._index.get(word)


def _stems(path: str = LEXICON) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            w, tag = line.split("\t")
            rows.append((w.strip(), tag.strip()))
    return rows


# Every closed class must survive truncation: a vocabulary without
# prepositions or punctuation cannot produce a sentence at any size.
CLOSED = ("DET", "PRON", "PRONO", "AUXS", "AUXP", "MODAL", "NEG", "PART",
          "PREP", "CONJ", "WH", "NUM", "PUNCT")
OPEN = ("N", "V", "ADJ", "ADV", "NPL", "VS", "VD", "VG", "ADJC")


def _expand(path: str) -> list[tuple[str, str]]:
    """Stems in rough frequency order, then their inflected forms."""
    stems = _stems(path)
    ranked: list[tuple[str, str]] = list(stems)
    for w, tag in stems:
        if tag == "N":
            ranked.append((_plural(w), "NPL"))
    for w, tag in stems:
        if tag == "V":
            ranked += [(_third(w), "VS"), (_past(w), "VD"), (_ing(w), "VG")]
    for w, tag in stems:
        if tag == "ADJ" and _comparative(w):
            ranked.append((_comparative(w), "ADJC"))
    seen, out = set(), []
    for w, t in ranked:
        if w not in seen:
            seen.add(w)
            out.append((w, t))
    return out


def build(size: int = 255, path: str = LEXICON) -> Vocab:
    """All closed-class words, then the open classes filled round-robin."""
    everything = _expand(path)
    if size >= len(everything):
        return Vocab(everything)
    picked = [wt for wt in everything if wt[1] in CLOSED][:size]
    pools = {t: [wt for wt in everything if wt[1] == t] for t in OPEN}
    while len(picked) < size and any(pools.values()):
        for tag in OPEN:
            if pools[tag] and len(picked) < size:
                picked.append(pools[tag].pop(0))
    rank = {wt: i for i, wt in enumerate(everything)}
    picked.sort(key=lambda wt: rank[wt])
    return Vocab(picked)


def from_lines(lines, size: int = 0) -> Vocab:
    """Imported vocab. Lines are ``word`` or ``word<TAB>TAG``; untagged -> N."""
    words, seen = [], set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        w = parts[0].strip()
        t = parts[1].strip().upper() if len(parts) > 1 else "N"
        if w in seen:
            continue
        seen.add(w)
        words.append((w, t if t in GLOSS else "N"))
        if size and len(words) >= size:
            break
    return Vocab(words)


def buckets(words, width: int = 255) -> list[tuple[str, list[tuple[str, str]]]]:
    """Split >255 words into labelled alphabetical buckets for a decode level."""
    ordered = sorted(words)
    out = []
    for i in range(0, len(ordered), width):
        chunk = ordered[i:i + width]
        label = f"{chunk[0][0]} ... {chunk[-1][0]}"
        out.append((label, chunk))
    return out


def bucket_criteria(groups) -> dict[str, str]:
    """Name real words in each bucket, or the choice is made blind."""
    return {label: "The next word is one of: " +
                   ", ".join(w for w, _ in chunk[:8]) + ("..." if len(chunk) > 8 else "")
            for label, chunk in groups}
