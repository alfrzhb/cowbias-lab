"""CLI entry point. Install the package with pip install -e '.[dev]' first."""

from cowbias.data.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
