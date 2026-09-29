"""Render every course figure: runs each ``figures/m*.py`` module's functions named ``fig_*``."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    names = sys.argv[1:]
    for path in sorted(HERE.glob("m*.py")):
        module = importlib.import_module(path.stem)
        for attr in sorted(dir(module)):
            if not attr.startswith("fig_"):
                continue
            if names and attr not in names and path.stem not in names:
                continue
            out = getattr(module, attr)()
            print(f"{path.stem}.{attr} -> {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
