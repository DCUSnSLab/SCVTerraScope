"""Thin wrapper around `ut-amrl/coda-devkit`'s `download_split.py`.

Clones (or reuses an existing checkout of) the CODa devkit, then invokes its
official downloader. Downloaded data lands under SCVTerraScope's
`data/coda_samples/` (override with `--dest`).

Why this wrapper exists:
- coda-devkit is the single source of truth for CODa download mechanics. We
  do NOT reimplement it; we only standardize the path and surface a `--help`
  consistent with our other CLI tools.
- The devkit assumes its own conda env (Python 3.8–3.9). This wrapper does
  not enforce that — users can pass `--python` to point at the right
  interpreter (e.g., a sibling conda env).

Usage:
    # Smallest split (recommended first run)
    python scripts/fetch_coda_samples.py --split tiny

    # A single sequence (~17GB for sequence 0)
    python scripts/fetch_coda_samples.py --sequence 0

    # Reuse a coda-devkit clone you already have
    python scripts/fetch_coda_samples.py --split tiny \\
        --coda-devkit-dir ~/repos/coda-devkit

    # Dry-run: print the commands without executing
    python scripts/fetch_coda_samples.py --split tiny --dry-run

See docs/runbooks/data_setup.md for the full procedure.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

CODA_DEVKIT_REPO = "https://github.com/ut-amrl/coda-devkit.git"
DEFAULT_DEST = Path("data/coda_samples")
DEFAULT_DEVKIT_DIR = Path("data/coda-devkit")
VALID_SPLITS = ("tiny", "small", "medium", "full")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--split",
        choices=VALID_SPLITS,
        help="CODa split to download. 'tiny' is the smallest and recommended for first runs.",
    )
    target.add_argument(
        "--sequence",
        type=int,
        metavar="N",
        help="Download a single sequence by id (0–21). One sequence is ~17GB.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Where downloaded data should land (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--coda-devkit-dir",
        type=Path,
        default=DEFAULT_DEVKIT_DIR,
        help=(
            "Path to an existing coda-devkit clone, or where to clone it if missing "
            f"(default: {DEFAULT_DEVKIT_DIR})."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python interpreter to invoke download_split.py with. Default: this "
            "interpreter. coda-devkit recommends Python 3.8–3.9 in their conda env — "
            "pass that interpreter here if their script fails on a newer Python."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    args = parser.parse_args(argv)

    if args.split is None and args.sequence is None:
        parser.error("must pass either --split or --sequence")
    if args.sequence is not None and not (0 <= args.sequence <= 21):
        parser.error("--sequence must be in [0, 21]")
    return args


def run(cmd: list[str], *, dry_run: bool, cwd: Path | None = None) -> None:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    cwd_label = f" (cwd={cwd})" if cwd else ""
    print(f"[fetch_coda_samples] $ {pretty}{cwd_label}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_coda_devkit(devkit_dir: Path, *, dry_run: bool) -> None:
    if (devkit_dir / ".git").exists():
        print(f"[fetch_coda_samples] coda-devkit already at {devkit_dir} — reusing.")
        return
    if devkit_dir.exists():
        # `.gitkeep` placeholder is fine; anything else looks like a stale
        # checkout we should not overwrite.
        non_placeholder = [p for p in devkit_dir.iterdir() if p.name != ".gitkeep"]
        if non_placeholder:
            raise SystemExit(
                f"refusing to clone into non-empty {devkit_dir} (no .git found). "
                "Pass --coda-devkit-dir to a fresh path or remove the directory."
            )
        # Remove the placeholder so `git clone` succeeds (it requires the
        # target path to be empty or non-existent).
        if not dry_run:
            (devkit_dir / ".gitkeep").unlink(missing_ok=True)
            devkit_dir.rmdir()
    devkit_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", CODA_DEVKIT_REPO, str(devkit_dir)], dry_run=dry_run)


def run_downloader(args: argparse.Namespace) -> None:
    download_script = args.coda_devkit_dir / "scripts" / "download_split.py"
    if not args.dry_run and not download_script.exists():
        raise SystemExit(
            f"expected {download_script} after clone — coda-devkit layout may have changed. "
            "Inspect the clone and update this wrapper."
        )
    args.dest.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, str(download_script.resolve()), "-d", str(args.dest.resolve())]
    if args.split is not None:
        cmd += ["-t", "split", "-sp", args.split]
    else:
        cmd += ["-t", "sequence", "-se", str(args.sequence)]
    # download_split.py is invoked from inside the coda-devkit checkout because it
    # may rely on relative paths to its own resources.
    run(cmd, dry_run=args.dry_run, cwd=args.coda_devkit_dir)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_coda_devkit(args.coda_devkit_dir, dry_run=args.dry_run)
    run_downloader(args)
    target_label = f"split={args.split}" if args.split else f"sequence={args.sequence}"
    print(f"[fetch_coda_samples] done — {target_label} → {args.dest}")


if __name__ == "__main__":
    main()
