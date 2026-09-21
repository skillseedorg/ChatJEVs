"""chatjevs command line. Everything it learns or produces goes into chatjevs.db."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time

from . import bench, db, vocab as vocab_mod
from .client import Jev, JevError, noul
from .net import ARCHS, Net


# ---- setup ---------------------------------------------------------------
def api_key(con, force: bool = False) -> str:
    key = None if force else (db.get(con, "api_key") or os.environ.get("TYPESAFE_API_KEY"))
    while not key:
        print("TypeSafe API key needed (console.typesafe.ai/keys). Stored in chatjevs.db.")
        key = getpass.getpass("API key: ").strip()
    db.put(con, "api_key", key)
    return key


def make_jev(con, args) -> Jev:
    return Jev(api_key(con), con, model=getattr(args, "model", "jev-latest"),
               use_cache=not getattr(args, "no_cache", False),
               workers=getattr(args, "workers", 8))


def make_vocab(con, arch) -> vocab_mod.Vocab:
    imported = db.load_vocab(con, db.get(con, "vocab", "builtin"))
    if imported and arch.vocab_size > 700:
        return vocab_mod.Vocab(imported[:arch.vocab_size])
    return vocab_mod.build(arch.vocab_size)


def report(jev: Jev, seconds: float) -> None:
    u = jev.usage
    print(f"\n{u['questions']} jev answers over {u['requests']} requests "
          f"({u['cached']} cached) | {u['in_tokens']}+{u['out_tokens']} tokens "
          f"| {seconds:.1f}s")


# ---- commands ------------------------------------------------------------
def cmd_init(con, args):
    key = api_key(con, force=True)
    jev = Jev(key, con, use_cache=False)
    try:
        answers = jev.ask("The cat sat on the mat.",
                          {"ok": noul("This text is written in English.")})
        print(f"connected. test answer: {answers['ok']['noul']}")
    except JevError as exc:
        print(f"key saved, but the test call failed: {exc}")
        return 1
    return 0


def cmd_info(con, args):
    print(f"{'arch':<6}{'jevs':>6}{'vocab':>8}{'ctx':>6}{'req/tok':>9}  layout")
    for name, a in ARCHS.items():
        if args.arch and args.arch != name:
            continue
        layout = (f"sem {a.semantic} syn {a.syntax} mem {a.memory} con {a.concepts} "
                  f"route {a.routing} plan {a.plan} dec {a.decoder}"
                  + (f" critic {a.critics}" if a.critics else "")
                  + (f" drafts {a.drafts}" if a.drafts else ""))
        print(f"{name:<6}{a.jevs:>6}{len(make_vocab(con, a)):>8}{a.window:>6}"
              f"{a.requests_per_token:>9}  {layout}")
        print(f"{'':>35}{a.note}")
    return 0


def cmd_gen(con, args):
    arch = ARCHS[args.arch]
    jev, vocab = make_jev(con, args), make_vocab(con, arch)
    net = Net(arch, jev, vocab, temperature=args.temp, top_p=args.top_p, seed=args.seed)
    run_id = db.start_run(con, "gen", args.arch, args.prompt)
    started = time.time()
    print(f"[{args.arch}: {arch.jevs} jevs, {len(vocab)} words]\n{args.prompt}", end="", flush=True)
    idx = 0

    def emit(word, dist, memory):
        nonlocal idx
        print(f" {word}", end="", flush=True)
        db.add_step(con, run_id, idx, word, {"memory": memory}, dist)
        idx += 1

    try:
        net.generate(args.prompt, args.n, on_token=emit, reply=not args.continue_)
    except JevError as exc:
        print(f"\nstopped: {exc}")
    print()
    if net.memory:
        print("memory:", json.dumps(net.memory))
    seconds = time.time() - started
    db.finish_run(con, run_id, " ".join(t["token"] for t in net.trace),
                  jev.usage, seconds, {"jevs": arch.jevs, "vocab": len(vocab)})
    report(jev, seconds)
    return 0


def _report_bench(res):
    lo, hi = res["ci"]
    print(f"\naccuracy {res['accuracy']:.2f}   chance {res['chance']:.2f}   "
          f"lift {res['lift']:+.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
    for cat, v in sorted(res["by_category"].items()):
        clo, chi = v["ci"]
        print(f"  {cat:<14}{v['accuracy']:.2f}  n={v['n']:<3} [{clo:.2f}, {chi:.2f}]")
    gen = res.get("generation") or {}
    if gen:
        print(f"\ngeneration {gen['overall']:.2f}   " +
              "  ".join(f"{k}={gen[k]:.2f}" for k in bench.CHECKS))


def cmd_bench(con, args):
    arch = ARCHS[args.arch]
    jev, vocab = make_jev(con, args), make_vocab(con, arch)
    tasks = bench.select(args.category, args.limit)
    prompts = () if args.no_gen else bench.PROMPTS[:args.gen or None]
    print(f"[{args.arch}: {arch.jevs} jevs] {len(tasks)} cloze tasks, "
          f"{len(prompts)} generation probes")
    res = bench.run(con, args.arch, jev, vocab, tasks, prompts, workers=args.workers)
    _report_bench(res)
    report(jev, res["seconds"])
    return 0


def cmd_audit(con, args):
    problems = bench.audit()
    for line in problems:
        print(" ", line)
    print(f"{len(bench.CLOZE)} items, {len(bench.CATEGORIES)} categories, "
          f"{len(problems)} problems")
    return 1 if problems else 0


def cmd_compare(con, args):
    tasks = bench.select(args.category, args.limit)
    prompts = () if args.no_gen else bench.PROMPTS[:args.gen or None]
    rows = {}
    for name in args.archs.split(","):
        arch = ARCHS[name]
        jev, vocab = make_jev(con, args), make_vocab(con, arch)
        print(f"\n[{name}: {arch.jevs} jevs] {len(tasks)} tasks")
        rows[name] = bench.run(con, name, jev, vocab, tasks, prompts,
                               workers=args.workers, verbose=args.verbose)
        r = rows[name]
        print(f"  accuracy {r['accuracy']:.2f} (lift {r['lift']:+.2f}) "
              f"| {r['usage']['requests']} requests "
              f"| {r['usage']['in_tokens'] + r['usage']['out_tokens']} tokens "
              f"| {r['seconds']:.0f}s")
    print("\n" + bench.table(rows))
    return 0


def cmd_runs(con, args):
    if args.id:
        row = con.execute("SELECT * FROM runs WHERE id=?", (args.id,)).fetchone()
        if not row:
            print("no such run")
            return 1
        print(json.dumps(dict(row), indent=2))
        for step in con.execute("SELECT idx,token,dist FROM steps WHERE run_id=? ORDER BY idx",
                                (args.id,)):
            dist = sorted(json.loads(step["dist"]).items(), key=lambda kv: -kv[1])[:4]
            print(f"  {step['idx']:>3} {step['token']:<12} " +
                  "  ".join(f"{w}={p:.2f}" for w, p in dist))
        for ev in con.execute("SELECT task,category,chosen,answer,correct FROM evals "
                              "WHERE run_id=?", (args.id,)):
            print(f"  {ev['task']:<4}{ev['category']:<12}{ev['chosen']:<10}"
                  f"{ev['answer']:<10}{'ok' if ev['correct'] else 'MISS'}")
        return 0
    print(f"{'id':>4} {'kind':<7}{'arch':<6}{'jevs':>5}{'req':>6}{'sec':>7}  result")
    for row in con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (args.limit,)):
        print(f"{row['id']:>4} {row['kind'] or '':<7}{row['arch'] or '':<6}"
              f"{row['jevs'] or 0:>5}{row['requests'] or 0:>6}{row['seconds'] or 0:>7.1f}  "
              f"{(row['output'] or '')[:60]}")
    return 0


def cmd_vocab(con, args):
    if args.import_path:
        with open(args.import_path, encoding="utf-8") as fh:
            v = vocab_mod.from_lines(fh)
        db.save_vocab(con, args.name, v.words)
        db.put(con, "vocab", args.name)
        print(f"imported {len(v)} words as {args.name!r} (now active)")
        return 0
    if args.use:
        if not db.load_vocab(con, args.use) and args.use != "builtin":
            print("no such vocab")
            return 1
        db.put(con, "vocab", args.use)
        print(f"active vocab: {args.use}")
        return 0
    active = db.get(con, "vocab", "builtin")
    print(f"builtin{'  *' if active == 'builtin' else '   '} {len(vocab_mod.build(10**6))} words")
    for row in con.execute("SELECT name,words FROM vocabs"):
        mark = "  *" if row["name"] == active else "   "
        print(f"{row['name']}{mark} {len(json.loads(row['words']))} words")
    return 0


def cmd_serve(con, args):
    from .server import serve
    serve(args.host, args.port)
    return 0


def cmd_key(con, args):
    api_key(con, force=True)
    print("stored")
    return 0


# ---- wiring --------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chatjevs", description="a language model made of Jev decisions")
    p.add_argument("--db", help="sqlite path (default ./chatjevs.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def shared(sp, arch_default="v1"):
        sp.add_argument("--arch", default=arch_default, choices=list(ARCHS))
        sp.add_argument("--model", default="jev-latest")
        sp.add_argument("--no-cache", action="store_true")
        sp.add_argument("--workers", type=int, default=8)

    sp = sub.add_parser("serve", help="run the ChatJEVs web UI")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--host", default="127.0.0.1")
    sp.set_defaults(fn=cmd_serve)

    sub.add_parser("init", help="store an API key and test it").set_defaults(fn=cmd_init)
    sub.add_parser("key", help="replace the stored API key").set_defaults(fn=cmd_key)

    sp = sub.add_parser("info", help="show architectures")
    sp.add_argument("--arch", choices=list(ARCHS))
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("gen", help="generate text")
    sp.add_argument("prompt")
    sp.add_argument("-n", type=int, default=12, help="max words")
    sp.add_argument("--temp", type=float, default=0.8)
    sp.add_argument("--top-p", type=float, default=0.95)
    sp.add_argument("--seed", type=int)
    sp.add_argument("--continue", dest="continue_", action="store_true",
                    help="continue the prompt instead of replying to it")
    shared(sp)
    sp.set_defaults(fn=cmd_gen)

    sp = sub.add_parser("bench", help="score one architecture")
    sp.add_argument("--category", choices=bench.CATEGORIES)
    sp.add_argument("--limit", type=int, default=0, help="tasks per category (0 = all)")
    sp.add_argument("--gen", type=int, default=0, help="generation probes (0 = all)")
    sp.add_argument("--no-gen", action="store_true", help="skip the generation half")
    shared(sp)
    sp.set_defaults(fn=cmd_bench)

    sub.add_parser("audit", help="check the benchmark for solvable-by-trick items"
                   ).set_defaults(fn=cmd_audit)

    sp = sub.add_parser("compare", help="score several architectures side by side")
    sp.add_argument("--archs", default="v0,v1,v2,v3")
    sp.add_argument("--category", choices=bench.CATEGORIES)
    sp.add_argument("--limit", type=int, default=0, help="tasks per category (0 = all)")
    sp.add_argument("--gen", type=int, default=0, help="generation probes (0 = all)")
    sp.add_argument("--no-gen", action="store_true", help="skip the generation half")
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--model", default="jev-latest")
    sp.add_argument("--no-cache", action="store_true")
    sp.add_argument("--workers", type=int, default=8)
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("runs", help="list or inspect stored runs")
    sp.add_argument("--id", type=int)
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(fn=cmd_runs)

    sp = sub.add_parser("vocab", help="list, import or select a vocabulary")
    sp.add_argument("--import", dest="import_path", metavar="FILE")
    sp.add_argument("--name", default="imported")
    sp.add_argument("--use")
    sp.set_defaults(fn=cmd_vocab)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.db:
        db.PATH = args.db          # the server opens its own connections
    con = db.connect(args.db)
    try:
        return args.fn(con, args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except JevError as exc:
        print(f"jev error: {exc}")
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
