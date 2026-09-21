# ChatJEVs

**[jevs.chat](https://jevs.chat)** — a language model with no language model in it.

Every word this system writes is chosen by a network of bounded decisions —
multiple choice, ratings, and yes/no probabilities. No component of it can
generate text.

[TypeSafe's Jev](https://docs.typesafe.ai) is a decision model. You hand it state
and a set of questions; it returns calibrated probabilities over the answers. It
has exactly three question types and none of them emit a token:

| | what it does | ceiling |
|---|---|---|
| **Choice** | pick one of N labelled options, with a probability for each | 255 options |
| **Score** | place something on a scale you define | 10 levels |
| **Noul** | one statement, one probability that it holds | — |

ChatJEVs samples each word from a Choice over candidate words. Everything else in
the architecture exists to make that one Choice a good one. Nothing is trained —
Jev is frozen. What changes between V0 and V5 is only the software around it.

> Can a network of calibrated decision models produce language when every unit
> only makes a bounded decision?

---

## Quick start

Python 3.10+. No dependencies.

```bash
python -m chatjevs serve        # then open http://127.0.0.1:8000
```

The app asks for your TypeSafe API key on first load, verifies it against the
API, and stores it in `chatjevs.db` next to the project. Everything else —
response cache, runs, per-word distributions, benchmark results — lands in the
same file. Nothing is written anywhere else.

Prefer the terminal?

```bash
python -m chatjevs init                      # store and test an API key
python -m chatjevs gen "what is a cat ?"     # reply, streaming
python -m chatjevs gen "the cat" --continue  # continue the prompt instead
python -m chatjevs bench --arch v2           # 40-task benchmark
python -m chatjevs compare --archs v0,v1,v2,v3
python -m chatjevs runs --id 3               # a run's words and distributions
python -m chatjevs vocab --import words.txt  # a larger vocabulary
```

---

## The app

**ChatJEVs** is a bootleg of a chat app you may recognise, except the model is
521 decision units arguing about grammar.

- **Chat** — send a message, watch the reply assemble a word at a time. Each
  response draws the network firing: every unit as a node, lit in the real order
  the engine consults them, with a pulse at the output each time a word lands.
  Expand the per-word trace to see the probability distribution behind every
  choice and the working-memory slots as they fill.
- **How it works** — the written explanation, with a live diagram of each
  version's own units firing in its real order.
- **System card** (`/writeup`) — a longer research note with charts. The measured
  results are read from your `chatjevs.db`; until you run a benchmark it says so
  rather than showing a number.

---

## The six versions

| | units | per word | vocab | what it adds |
|---|---|---|---|---|
| **V0** | 10 | 10 | 255 | nothing — the control |
| **V1** | 32 | 30 | 255 | memory slots, recurrence, a plan |
| **V2** | 100 | 98 | 1709 | a concept bank, hierarchical decoder |
| **V3** | 253 | 251 | 1709 | routing: gated banks, attention over memory |
| **V4** | 382 | 380 | 1709 | eight critics judge every candidate word |
| **V5** | 521 | 380 | 1709 | three drafts, a judging panel, a repair pass |

**V0–V3 improve the inputs.** V1 tracks subject/action/object/location/time as
slots and feeds the previous step's most confident answers back in, making it
recurrent. V2 adds concept units — small questions about animacy, size, motion,
polarity — so meaning lives in a pattern across units rather than in any one of
them. V3 adds routing units that decide which banks run at all and which memory
slot the decoder attends to: attention, decided rather than computed.

All four share one flaw. Every word is a single 255-way guess, committed
instantly, with no lookahead and no way back.

**V4 judges before it commits.** The decoder proposes its six best candidates; a
panel of eight critics scores each one in a single request — grammatical here,
would a fluent writer write it, relevant, advances the reply, asserts anything
false, still a way to finish the sentence, repetitive, right tone. The verdict is
blended with the decoder's own probability. The bet is that *"is `cat` right
here?"* is what a decision model is built for, and *"pick one of 255"* is not.

**V5 deliberates.** Three drafts at rising temperature written by the cheap V1
architecture, a panel that scores all three and votes head to head, then a repair
pass marking every position keep / replace / cut. The only version that can fix
its own second word.

---

## One word

```
plan      speech act and length              once per reply
route     which banks are worth running      V3+
banks     meaning · grammar · memory · concepts
memory    slots updated in ordinary code, outside Jev
decode    form → bucket → word               + an <end> option
critics   judge the top candidates           V4+
sample    a word falls out of the distribution
```

Questions batched into one request are answered independently and in parallel —
documented at about **12× cheaper and 10× faster** than asking separately. That
is why 521 units cost three to six requests per word, not 521. `client.py` splits
anything over 64 questions across parallel requests automatically.

### What the decoder offers

255 slots, filled in priority order:

1. words matching the role and inflection the syntax units settled on
2. **words copied from the conversation** — with a small vocabulary this is most
   of what keeps a reply on topic
3. frequent words, filling the rest, so a wrong guess upstream is recoverable
4. `<end>`, so the decoder can decide the reply is finished

Each option is described by naming it — `"The next English word is 'cat' (noun,
singular)"`. An earlier build described options by part of speech, which made all
34 nouns in a candidate list read identically; the decoder was choosing blind.
This, the copy-from-prompt candidates and the plan stage are all lifted from
[jev-llm](https://github.com/afanjul/jev-llm), which got there first.

---

## Benchmark

Two instruments, because forced choice alone cannot see what V4 and V5 do.

**Cloze** — 90 minimal pairs across nine categories: `agreement`, `attraction`,
`binding`, `case`, `tense`, `quantifier`, `recall`, `selection`,
`plausibility`. Every distractor fills the same syntactic slot as the answer, so
no item is won by picking the only auxiliary or the only noun. In the recall
items every option appears in the passage, so copying decides nothing. Option
counts differ, so results are reported against each item's own chance level with
a 95% interval — ten items per category means one flip moves a score ten points,
and the interval says so out loud.

**Generation** — replies to eight probes, graded by ordinary Python with no Jev
in the scoring loop: did it produce anything, does it stutter, does it repeat,
does the verb agree with the noun, is the article right before a vowel, is there
a verb at all, did it merely echo the prompt, did it stop by itself. This is the
half that measures what the architecture is for.

The suite audits itself:

```bash
python -m chatjevs audit     # fails if any item is solvable by a cheap trick
```

> The previous suite had 40 items and **23 of them could be answered by picking
> the only option of the right part of speech** — including every item in the
> category it claimed was hardest. It was measuring lazy distractors. The audit
> exists so that cannot happen again.

Public LLM benchmarks do not measure this system. **Jev is a System One model: it
decides over state you supply, it does not store facts.** A benchmark whose
answer is not in the prompt gives it nothing to decide between, so MMLU, ARC and
friends land at chance — and the 521 units sit inert while a single Choice does
the work. Generative suites (GSM8K, HumanEval, IFEval) would measure the
architecture, but a ~1,700-word vocabulary with no digits and no code tokens
makes them impossible rather than merely hard. The system card works through the
numbers.

---

## Layout

```
chatjevs/
  client.py        Jev transport: batching, cache, retries, usage accounting
  banks.py         the units, as ordered tables — semantic, syntax, memory,
                   concept, routing, critics, draft judges
  net.py           the engine: state, gating, memory, decoder, critics,
                   deliberation, and the six architectures
  bench.py         tasks and scoring
  vocab.py         ~820 stems inflected to ~1709 words; imports anything larger
  db.py            SQLite
  server.py        local HTTP + SSE
  cli.py           the command line
  web/index.html   ChatJEVs
  web/writeup.html the system card
  web/brain.js     the network visualisation, shared by both
data/lexicon.txt
```

An architecture takes the first *N* entries of each bank, so V0 gets the most
load-bearing questions and V5 gets the whole tail. Adding a unit is one line in
`banks.py`.

---

## Limitations

- **It replies; it does not continue your prompt.** Your message is state and the
  network writes an answer word by word. `--continue` gets the old behaviour,
  which is what the benchmark uses.
- **The vocabulary is ~1,709 words**, not the 5k/20k V2 and V3 can address. Those
  are decoder ceilings, not what ships. V0 and V1 are held to 255 by design, so
  they sound worse for that reason alone. `chatjevs vocab --import words.txt` (one
  word per line, optionally `word<TAB>TAG`) raises it.
- **Replies are short by construction** — capped at 20 words, and the plan unit
  often asks for fewer. The first word is always produced, so a prompt can never
  come back empty.
- **V4 and V5 are expensive.** ~380 answered questions per word for V4; V5 adds
  three full drafts plus judging and repair. V2 is the sensible default.
- **Responses are cached** by `(model, state, questions)`, so re-running a
  benchmark is free. `--no-cache` forces fresh calls.
- **There is no training loop.** There are no weights to move. Improving the
  system means editing a bank and re-running the benchmark — that is the entire
  optimisation loop.
- **More units is not obviously better.** Separate Jev calls are not independent
  neurons, and the concept banks may be adding noise. The benchmark settles it,
  not the unit count.
- Jev is a decision model, not a generative one. Whether this produces language
  is the experiment, not the premise.
