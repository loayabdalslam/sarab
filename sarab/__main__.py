"""Enables `python -m sarab ...` to invoke the CLI (same as the `sarab` console script)."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
