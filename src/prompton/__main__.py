"""``python -m prompton`` - fetch a use-case document and write it to a file.

Run this in CI on every build and commit the result as the bundle, so a brand-new container can
load use cases before it has ever reached PromptOn::

    python -m prompton export --out app/prompton/use-cases.production.json

On failure it exits non-zero and leaves any existing file untouched.
"""

from __future__ import annotations

import argparse
import json
import sys

from ._version import VERSION
from .client import PromptOn
from .errors import PromptOnError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prompton", description=__doc__)
    parser.add_argument("--version", action="version", version=f"prompton-sdk {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="fetch the use-case document and write it to a file")
    export.add_argument("--out", required=True, help="where to write the use-case JSON")
    export.add_argument("--environment", default=None, help="environment (default production)")
    export.add_argument("--host", default=None, help="PromptOn host (default $PTN_HOST)")
    export.add_argument("--api-key", default=None, help="runtime key (default $PTN_API_KEY)")

    info = sub.add_parser("info", help="print where the current use-case document came from")
    info.add_argument("--environment", default=None)
    info.add_argument("--host", default=None)
    info.add_argument("--api-key", default=None)

    args = parser.parse_args(argv)

    client = PromptOn(
        api_key=args.api_key,
        host=args.host,
        environment=args.environment,
        poll=False,
        disk_cache=False,
    )
    try:
        if args.command == "export":
            client.refresh(force=True)
            path = client.export_use_cases(args.out)
            print(f"wrote {path}")
            return 0
        client.refresh(force=True)
        print(json.dumps(client.use_cases_info(), indent=2, default=str))
        return 0
    except PromptOnError as error:
        print(f"prompton: {error}", file=sys.stderr)
        return 1
    finally:
        client.close(timeout=1.0)


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
