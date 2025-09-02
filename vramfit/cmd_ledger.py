"""The ledger subcommand."""
from __future__ import annotations

from vramfit import ledger_diff
from vramfit.ledger import KEY_COLUMNS, Ledger


def add_parser(sub):
    p = sub.add_parser("ledger", help="measurements I have taken")
    p.add_argument("action", choices=("list", "show", "diff"))
    p.add_argument("--db")
    for col in KEY_COLUMNS:
        p.add_argument(f"--{col.replace('_', '-')}")
    p.add_argument("--all-fields", action="store_true")
    p.set_defaults(fn=run)
    return p


def key_from(args) -> dict:
    out = {}
    for col in KEY_COLUMNS:
        val = getattr(args, col, None)
        if val is None:
            continue
        out[col] = int(val) if col == "context" else val
    return out


def run(args) -> int:
    with Ledger(args.db) as led:
        key = key_from(args)

        if args.action == "list":
            rows = led.keys()
            if not rows:
                print("ledger is empty")
                return 0
            print(f"{'model':28} {'quant':10} {'ctx':>7} "
                  f"{'backend':10} {'gpu':24} {'n':>3}")
            for r in rows:
                print(f"{r['model'][:28]:28} {r['quant'][:10]:10} "
                      f"{r['context']:7} {r['backend'][:10]:10} "
                      f"{r['gpu'][:24]:24} {r['n']:3}")
            return 0

        rows = led.history(**key)
        if not rows:
            print("no rows match that key")
            return 1

        if args.action == "show":
            for r in rows:
                print(f"{r['taken_at']}  decode "
                      f"{r['decode_tok_s'] or 0:7.1f} tok/s  "
                      f"({(r['decode_frac_peak'] or 0):.0%} of peak)  "
                      f"prefill {r['prefill_tok_s'] or 0:8.1f} tok/s  "
                      f"driver {r['driver']}  "
                      f"build {r['backend_version']}")
            return 0

        if len(rows) < 2:
            print("need two measurements under one key to diff; "
                  f"have {len(rows)}")
            return 1
        print(ledger_diff.render(rows[-2], rows[-1],
                                 all_fields=args.all_fields))
        return 0
