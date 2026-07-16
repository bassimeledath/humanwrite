#!/usr/bin/env python3
"""Append-only ledger for preregistration + run registry (STUB-complete
enough to preregister and query; run-lifecycle fields are written by the gpu
wrapper). JSONL, one object per line, append-only. Never rewrite history."""
import argparse, json, os, sys, time, hashlib

PATH = os.path.join(os.path.dirname(__file__), "ledger.jsonl")


def _append(obj):
    obj["ts"] = time.time()
    with open(PATH, "a") as f:
        f.write(json.dumps(obj) + "\n")
    # cheap backup
    try:
        with open(PATH) as src, open(PATH + ".bak", "w") as dst:
            dst.write(src.read())
    except OSError:
        pass


def _read():
    if not os.path.exists(PATH):
        return []
    return [json.loads(l) for l in open(PATH) if l.strip()]


def add(args):
    """Preregister a comparison OR add a run entry."""
    if args.hypothesis is not None:
        _append({"kind": "prereg", "comparison": args.comparison,
                 "hypothesis": args.hypothesis, "arms": args.arms or "",
                 "status": "open"})
        print(f"preregistered {args.comparison}")
    else:
        _append({"kind": "run", "run_id": args.run_id,
                 "comparison": args.comparison,
                 "config_hash": args.config_hash, "git_sha": args.git_sha,
                 "budget_class": args.budget_class, "seed": args.seed,
                 "data_split_hash": args.data_split_hash,
                 "status": "launched"})
        print(f"run entry added {args.run_id}")


def update(args):
    _append({"kind": "run_update", "run_id": args.run_id,
             "status": args.status, "cost": args.cost,
             "accel_seconds": args.accel_seconds, "tokens": args.tokens,
             "metrics_ptr": args.metrics_ptr})


def query(args):
    rows = _read()
    if args.comparison:
        rows = [r for r in rows if r.get("comparison") == args.comparison]
    if args.open:
        preregs = [r for r in rows if r.get("kind") == "prereg" and r.get("status") == "open"]
        print(json.dumps(preregs))
        return
    print(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser(prog="ledger")
    s = p.add_subparsers(dest="cmd", required=True)

    pa = s.add_parser("add")
    pa.add_argument("--comparison", required=True)
    pa.add_argument("--run-id", dest="run_id", default=None)
    pa.add_argument("--hypothesis", default=None, help="present => preregistration")
    pa.add_argument("--arms", default=None)
    pa.add_argument("--config-hash", dest="config_hash", default=None)
    pa.add_argument("--git-sha", dest="git_sha", default=None)
    pa.add_argument("--budget-class", dest="budget_class", default=None)
    pa.add_argument("--seed", type=int, default=None)
    pa.add_argument("--data-split-hash", dest="data_split_hash", default=None)
    pa.set_defaults(f=add)

    pu = s.add_parser("update")
    pu.add_argument("--run-id", dest="run_id", required=True)
    pu.add_argument("--status", required=True)
    pu.add_argument("--cost", type=float, default=None)
    pu.add_argument("--accel-seconds", dest="accel_seconds", type=float, default=None)
    pu.add_argument("--tokens", type=int, default=None)
    pu.add_argument("--metrics-ptr", dest="metrics_ptr", default=None)
    pu.set_defaults(f=update)

    pq = s.add_parser("query")
    pq.add_argument("--comparison", default=None)
    pq.add_argument("--open", action="store_true")
    pq.set_defaults(f=query)

    a = p.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
