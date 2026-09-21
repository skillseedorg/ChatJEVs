"""The Jev units themselves, as ordered tables.

Each entry is (key, builder). An architecture takes the first N of a bank, so
V0 gets the most load-bearing questions and V3 gets the whole tail.
"""
from __future__ import annotations

from .client import choice, noul, score

SLOTS = ["SUBJECT", "ACTION", "OBJECT", "LOCATION", "TIME", "MODIFIER", "RECIPIENT", "TOPIC"]
ROLE_OPTIONS = {
    "DETERMINER": "an article or quantifier before a noun",
    "NOUN": "a noun naming a thing, person or place",
    "PRONOUN": "a pronoun standing in for a noun",
    "VERB": "a main verb",
    "AUXILIARY": "a helping verb, modal, negation or the particle 'to'",
    "ADJECTIVE": "a word describing a noun",
    "ADVERB": "a word modifying a verb or adjective",
    "PREPOSITION": "a word introducing a phrase of place, time or manner",
    "CONJUNCTION": "a word joining clauses or phrases",
    "QUESTION_WORD": "a wh-word starting a question",
    "PUNCTUATION": "a full stop, comma or other mark",
    "END": "the text is finished and nothing should follow",
}
FOCUSES = ["the grammatical subject of the current sentence",
           "the word that should come next",
           "the situation the text describes as a whole",
           "the most recently written word",
           "what the user is actually asking about",
           "the sentence this reply is building toward"]


def _c(instr, opts):
    return lambda: choice(instr, opts)


def _n(text):
    return lambda: noul(text)


def _s(instr, levels):
    return lambda: score(instr, levels)


def _labels(*opts):
    return {o: o.replace("_", " ") for o in opts}


SEMANTIC = [
    ("entity", _c("What kind of entity is the text about right now?", {
        "person": "a human being", "animal": "a non-human creature",
        "object": "a physical thing", "place": "a location or space",
        "abstract": "an idea, feeling or quality", "none": "no entity is in play yet"})),
    ("action", _c("What kind of action is in play?", {
        "movement": "going, running, travelling", "rest": "being, staying, existing",
        "creation": "making or building", "destruction": "breaking or removing",
        "communication": "saying, asking, telling", "perception": "seeing, hearing, feeling",
        "possession": "having, giving, taking", "none": "no action yet"})),
    ("time_frame", _c("What time frame does the text describe?", {
        "past": "already happened", "present": "happening now",
        "future": "yet to happen", "timeless": "a general truth"})),
    ("sentence_complete", _n("The text so far is already a complete grammatical sentence.")),
    ("quantity", _c("How many of the main thing are involved?", {
        "one": "exactly one", "several": "a small group", "many": "a large group",
        "uncountable": "a mass noun like water", "unknown": "not determinable"})),
    ("speaker", _c("Whose point of view is the text written from?", {
        "first_person": "I or we", "second_person": "you",
        "third_person": "he, she, it or they", "none": "no clear viewpoint"})),
    ("relation", _c("What relation is being expressed?", {
        "agent_action": "someone doing something", "action_object": "something being acted on",
        "possession": "owning or having", "location": "being somewhere",
        "description": "a quality of something", "none": "none of these"})),
    ("intent", _c("What is the text doing?", {
        "describe": "describing a scene or state", "narrate": "telling what happened",
        "ask": "asking a question", "request": "asking for something",
        "command": "telling someone to act"})),
    ("definiteness", _c("Is the thing being discussed already known to the reader?", {
        "definite": "already introduced or unique", "indefinite": "newly introduced",
        "generic": "a class of things in general"})),
    ("physicality", _c("What domain does the content belong to?",
                       _labels("physical", "mental", "social", "abstract"))),
    ("sentiment", _c("What is the emotional colour of the text?",
                     _labels("positive", "negative", "neutral"))),
    ("certainty", _s("How certain is the claim in the text?",
                     ["pure speculation", "hedged", "stated as fact"])),
    ("participants", _c("How many participants are involved?",
                        _labels("none", "one", "two", "three_or_more"))),
    ("new_information", _n("The next word should introduce something new rather than "
                           "continue what is already established.")),
    ("topic_shift", _n("The text has just changed topic.")),
    ("concreteness", _s("How concrete is the subject matter?",
                        ["fully abstract", "mixed", "fully concrete"])),
    ("location_kind", _c("What kind of location is relevant?",
                         _labels("indoor", "outdoor", "none", "unknown"))),
    ("plurality_focus", _c("Is the thing in focus singular, plural or a mass?",
                           _labels("singular", "plural", "mass", "unknown"))),
    ("salience", _s("How strongly does one entity dominate the text?",
                    ["no clear focus", "one among several", "one dominant entity"])),
    ("discourse", _c("How does the current clause relate to what came before?",
                     _labels("statement", "continuation", "contrast", "cause", "result"))),
    ("specificity", _s("How specific is the language?",
                       ["very general", "moderate", "very specific"])),
    ("register", _s("How formal is the text?",
                    ["casual speech", "neutral", "formal writing"])),
    ("animacy_subject", _c("Is the grammatical subject alive?",
                           _labels("animate", "inanimate", "unknown"))),
    ("causality", _c("Is a cause-and-effect link being made?",
                     _labels("cause", "effect", "both", "none"))),
]

SYNTAX = [
    ("next_role", _c("What grammatical element must come next for the text to stay correct?",
                     ROLE_OPTIONS)),
    ("subject_number", _c("Is the subject that controls verb agreement singular or plural? "
                          "Look back to the head noun of the subject, not the nearest noun.",
                          _labels("singular", "plural", "unknown"))),
    ("tense", _c("What tense is the current sentence in?",
                 _labels("past", "present", "future", "none"))),
    ("clause_stage", _c("Where is the text inside the current clause?", {
        "before_subject": "nothing of the clause is written yet",
        "in_subject": "the subject noun phrase is unfinished",
        "after_subject": "the subject is complete and a verb is due",
        "in_verb": "the verb group is unfinished",
        "after_verb": "the verb is complete and a complement may follow",
        "in_object": "the object noun phrase is unfinished",
        "after_object": "the clause is complete"})),
    ("verb_form", _c("What form must the next verb take?", {
        "base": "plural or imperative form, like 'run'",
        "third_singular": "form ending in -s, like 'runs'",
        "past": "past tense form, like 'ran'",
        "progressive": "-ing form, like 'running'",
        "infinitive": "after 'to'", "none": "no verb is due"})),
    ("needs_determiner", _n("The next noun still needs a determiner such as 'the' or 'a' "
                            "before it.")),
    ("mood", _c("What kind of sentence is this?",
                _labels("declarative", "question", "command", "exclamation"))),
    ("expects_object", _n("The verb already written requires an object to follow.")),
    ("polarity", _c("Is the clause positive or negated?", _labels("positive", "negative"))),
    ("open_phrase", _c("Which phrase is currently open and unfinished?",
                       _labels("none", "noun_phrase", "verb_phrase",
                               "prepositional_phrase", "clause"))),
    ("person", _c("What grammatical person is the subject?",
                  _labels("first", "second", "third"))),
    ("punctuation_next", _n("A punctuation mark should come next.")),
    ("agreement_source", _c("Which earlier word controls agreement for the verb that is due?", {
        "head_noun": "the head noun of the subject phrase",
        "nearest_noun": "the noun closest to the gap",
        "pronoun": "a pronoun subject", "none": "no verb is due"})),
    ("depth", _c("Is the text inside a main clause or a subordinate one?",
                 _labels("main", "subordinate", "relative"))),
    ("aspect", _c("What aspect does the verb group have?",
                  _labels("simple", "progressive", "perfect", "none"))),
    ("voice", _c("Is the clause active or passive?", _labels("active", "passive"))),
    ("length_pressure", _s("How much more should this sentence contain?",
                           ["it should end now", "a little more", "much more"])),
    ("coordination", _n("The next word should join two clauses or phrases.")),
    ("modifier_slot", _n("A describing word would fit naturally in the next position.")),
    ("comparative", _n("A comparison is being set up.")),
    ("question_type", _c("If this is a question, what kind is it?",
                         _labels("yes_no", "wh", "none"))),
    ("repetition", _n("The text has started repeating itself.")),
    ("last_word_role", _c("What grammatical element was the last word written?", ROLE_OPTIONS)),
    ("well_formed", _n("The text so far is grammatically well formed.")),
]

_SLOT_OPTS = {s: s.lower() for s in SLOTS}
_STATUS = {"KEEP": "leave the stored value alone",
           "REPLACE": "overwrite it with the latest word", "CLEAR": "empty it"}

MEMORY = [
    ("slot_of_last", _c("Which working-memory slot does the most recent word belong to?",
                        dict(_SLOT_OPTS, NONE="it fills no slot"))),
    ("focus", _c("Which slot is the text focused on right now?", _SLOT_OPTS)),
    ("referent", _c("If a pronoun was just used, which slot does it refer back to?",
                    dict(_SLOT_OPTS, NONE="no pronoun in play"))),
    ("recall_target", _c("Which slot holds the answer to the question the text is asking?",
                         dict(_SLOT_OPTS, NONE="no question is being asked"))),
    ("sentence_boundary", _n("A sentence just ended, so short-lived slots may be cleared.")),
    ("new_entity", _n("A new entity was just introduced that memory should record.")),
    ("memory_conflict", _n("The stored memory contradicts what the text now says.")),
    ("carryover", _s("How much of the stored memory still applies?",
                     ["none of it", "some of it", "all of it"])),
]
MEMORY += [(f"{s.lower()}_status",
            _c(f"What should happen to the {s.lower()} slot?", _STATUS)) for s in SLOTS]
MEMORY += [(f"{s.lower()}_active",
            _n(f"The {s.lower()} slot is relevant to choosing the next word.")) for s in SLOTS]

CONCEPT_DIMS = [
    ("animacy", ["living", "nonliving", "abstract", "unclear"]),
    ("size", ["tiny", "small", "medium", "large", "unclear"]),
    ("polarity", ["good", "bad", "neutral"]),
    ("motion", ["moving", "still", "unclear"]),
    ("agency", ["acts", "is_acted_on", "neither"]),
    ("concreteness", ["concrete", "abstract", "unclear"]),
    ("temporality", ["past", "present", "future", "timeless"]),
    ("number", ["one", "few", "many", "mass"]),
    ("definiteness", ["known", "new", "generic"]),
    ("humanness", ["human", "animal", "nonhuman", "unclear"]),
    ("naturalness", ["natural", "artificial", "unclear"]),
    ("function", ["tool", "container", "surface", "none"]),
    ("emotion", ["calm", "excited", "upset", "none"]),
    ("speed", ["fast", "slow", "static", "unclear"]),
    ("intensity", ["weak", "moderate", "strong"]),
    ("social_role", ["family", "work", "stranger", "none"]),
    ("containment", ["inside", "outside", "on_top", "none"]),
    ("direction", ["toward", "away", "none"]),
    ("causal_role", ["causes", "is_caused", "neither"]),
    ("possession", ["owns", "is_owned", "neither"]),
    ("visibility", ["visible", "hidden", "unclear"]),
    ("edibility", ["edible", "inedible", "unclear"]),
    ("hardness", ["soft", "hard", "unclear"]),
    ("age", ["young", "old", "unclear"]),
    ("distance", ["near", "far", "unclear"]),
    ("frequency", ["once", "repeated", "habitual", "unclear"]),
    ("duration", ["brief", "extended", "unclear"]),
    ("necessity", ["required", "optional", "forbidden", "none"]),
    ("ability", ["able", "unable", "unclear"]),
    ("permission", ["allowed", "not_allowed", "none"]),
    ("comparison", ["more", "less", "equal", "none"]),
    ("part_whole", ["part", "whole", "neither"]),
    ("material", ["solid", "liquid", "gas", "none"]),
    ("shape", ["round", "flat", "long", "unclear"]),
    ("texture", ["smooth", "rough", "unclear"]),
    ("sound", ["loud", "quiet", "silent", "unclear"]),
    ("light", ["bright", "dark", "unclear"]),
    ("weather", ["clear", "wet", "cold", "none"]),
    ("danger", ["safe", "risky", "unclear"]),
    ("formality", ["casual", "neutral", "formal"]),
]

ROUTING = [
    ("next_action", _c("What should the network do next?", {
        "GENERATE_HEAD": "emit the main word of the current phrase",
        "CONTINUE_PHRASE": "emit a word continuing the open phrase",
        "CLOSE_PHRASE": "emit a word that closes the open phrase",
        "END_SENTENCE": "emit punctuation and finish",
        "RECALL": "answer from working memory instead of continuing"})),
    ("attend_slot", _c("Which memory slot matters most for choosing the next word?",
                       dict(_SLOT_OPTS, NONE="none of them"))),
    ("attend_window", _c("How much of the text is relevant to the next word?", {
        "last_3": "only the last few words", "last_8": "the current clause",
        "whole_context": "the whole passage"})),
    ("dominant_signal", _c("Which kind of constraint should decide the next word?", {
        "syntax": "grammatical form", "semantics": "meaning of the situation",
        "memory": "what was established earlier",
        "context": "the literal preceding words"})),
    ("run_semantics", _n("Re-reading the meaning of the text would change what comes next.")),
    ("run_memory", _n("Working memory needs updating before the next word.")),
    ("run_concepts", _n("Fine-grained conceptual features are needed to choose the next word.")),
    ("need_recall", _n("The text is asking about something stated earlier.")),
    ("need_detail", _n("The next word should add detail rather than move the sentence on.")),
    ("repetition_risk", _n("Continuing the obvious way would repeat earlier text.")),
    ("openness", _s("How many different words would fit the next position?",
                    ["only one", "a handful", "very many"])),
    ("risk", _s("How likely is the next choice to break the sentence?",
                ["safe", "uncertain", "fragile"])),
    ("clause_pressure", _s("How close is the current clause to being complete?",
                           ["just started", "midway", "nearly done"])),
    ("coherence", _s("How coherent is the text so far?", ["incoherent", "loose", "tight"])),
    ("commit", _n("The network should commit to the highest-probability word rather "
                  "than explore.")),
    ("concept_focus", _c("What should the concept units describe?",
                         {f"focus_{i}": f for i, f in enumerate(FOCUSES)})),
]


def concept_questions(count: int) -> dict:
    """count units = CONCEPT_DIMS cycled across FOCUSES."""
    out, per = {}, len(CONCEPT_DIMS)
    for i in range(count):
        dim, opts = CONCEPT_DIMS[i % per]
        focus = FOCUSES[(i // per) % len(FOCUSES)]
        out[f"c{i // per}_{dim}"] = choice(f"Considering {focus} - {dim}?", _labels(*opts))
    return out


def take(bank, count: int) -> dict:
    return {key: build() for key, build in bank[:count]}


# --- V4: judge candidates instead of trusting one 255-way guess -------------
# Jev is a decision model, so scoring "is this word right here?" sits closer to
# what it does well than "pick one of 255". Weights say which critics decide.
CRITIC_DIMS = [
    ("grammar", 1.4, "Writing '{w}' in the gap keeps the text grammatical English."),
    ("natural", 1.2, "A fluent English writer would actually write '{w}' at this point."),
    ("relevant", 1.2, "'{w}' fits what the text is about."),
    ("progress", 1.0, "'{w}' moves the text toward a complete, useful statement."),
    ("truthful", 1.0, "Writing '{w}' here does not assert anything false."),
    ("openable", 0.9, "After '{w}' there is still a natural way to finish the sentence."),
    ("fresh", 0.8, "'{w}' does not repeat something already said."),
    ("register", 0.6, "'{w}' matches the tone of the surrounding text."),
]


def critic_questions(candidates) -> dict:
    """One noul per (candidate, dimension) - all of it in a single request."""
    return {f"c{i}_{dim}": noul(text.format(w=word))
            for i, word in enumerate(candidates)
            for dim, _, text in CRITIC_DIMS}


def critic_scores(candidates, answers: dict) -> dict[str, float]:
    total = sum(weight for _, weight, _ in CRITIC_DIMS)
    out = {}
    for i, word in enumerate(candidates):
        acc = 0.0
        for dim, weight, _ in CRITIC_DIMS:
            acc += weight * float((answers.get(f"c{i}_{dim}") or {}).get("noul") or 0.0)
        out[word] = acc / total
    return out


# --- V5: judge whole drafts, then repair the winner -------------------------
DRAFT_DIMS = [
    ("answers", 1.4, "Draft {i} actually answers or responds to the user's message."),
    ("grammatical", 1.3, "Draft {i} is grammatical English from beginning to end."),
    ("complete", 1.1, "Draft {i} is finished: it does not stop mid-sentence."),
    ("ontopic", 1.1, "Draft {i} stays on the topic of the user's message."),
    ("natural", 1.0, "Draft {i} reads like something a person would write."),
    ("truthful", 1.0, "Draft {i} asserts nothing false."),
    ("concise", 0.7, "Draft {i} is no longer than it needs to be."),
    ("fresh", 0.7, "Draft {i} does not repeat itself."),
]

REVISE = {"KEEP": "this word is right; leave it alone",
          "REPLACE": "this word is wrong or clumsy; a better word belongs here",
          "CUT": "this word should be deleted entirely"}


def draft_questions(count: int) -> dict:
    return {f"d{i}_{dim}": noul(text.format(i=i + 1))
            for i in range(count) for dim, _, text in DRAFT_DIMS}


def draft_scores(count: int, answers: dict) -> list[float]:
    total = sum(weight for _, weight, _ in DRAFT_DIMS)
    out = []
    for i in range(count):
        acc = 0.0
        for dim, weight, _ in DRAFT_DIMS:
            acc += weight * float((answers.get(f"d{i}_{dim}") or {}).get("noul") or 0.0)
        out.append(acc / total)
    return out


def revise_questions(words) -> dict:
    return {f"w{i}": choice(
        f"In the draft, position {i + 1} holds the word '{word}'. What should happen to it?",
        REVISE) for i, word in enumerate(words)}
