"""Command line: ``cytherea run <config> | resume <config> | report <store>`` (Task 13).

``run`` starts a config into a new store, ``resume`` continues an
interrupted one (only missing keys run; the config must hash the same),
``report`` prints a JSON summary of a store. Exit status 0 on success, 2 on
a config, resume or usage error (message on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys

from pydantic import ValidationError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cytherea", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, what in (("run", "start a run from a config file"), ("resume", "continue an interrupted run")):
        s = sub.add_parser(name, help=what)
        s.add_argument("config", help="YAML config file")
    s = sub.add_parser("report", help="summarise a record store")
    s.add_argument("store", help="SQLite store path")
    return p


def main(argv: list[str] | None = None) -> int:
    from cytherea.config import dispatch, load_config, report
    from cytherea.config.runners import ResumeError

    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            out = report(args.store)
        else:
            cfg = load_config(args.config)
            out = dispatch(cfg, resume=args.command == "resume")()
    except ValidationError as exc:
        print(f"cytherea: invalid config:\n{exc}", file=sys.stderr)
        return 2
    except (ResumeError, FileNotFoundError, NotImplementedError) as exc:
        print(f"cytherea: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
