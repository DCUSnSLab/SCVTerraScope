"""`python -m scvterrascope.gui` and `scvterrascope-monitor` entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="scvterrascope-monitor")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
        help="Path to YAML config (default: configs/default.yaml).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path.")
    parser.add_argument("--device", default=None, help="auto / cuda / cuda:0 / cpu (overrides config).")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Local imports keep `python -m scvterrascope.gui --help` cheap (no Qt/torch).
    from PyQt6.QtWidgets import QApplication

    from scvterrascope.gui.config import load_config
    from scvterrascope.gui.main_window import MainWindow

    cfg = load_config(args.config)
    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint
    if args.device is not None:
        cfg.device = args.device

    app = QApplication(sys.argv)
    win = MainWindow(cfg)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
