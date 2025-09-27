#!/usr/bin/env python3

# fetch_dataset.py
# Usage:
#   python fetch_dataset.py [dataset] [config] [out_dir] [--force]
# Examples:
#   python fetch_dataset.py glue mrpc
#   python fetch_dataset.py imdb               # no config
#   python fetch_dataset.py glue mrpc /tmp/glue-mrpc --force

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
from datasets import load_dataset

def find_repo_root(start: Optional[Path] = None) -> Path:
    """Find repo root via `git rev-parse` or a `.git` directory; fall back to script dir."""
    start = start or Path(__file__).resolve().parent
    # Try git
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start, stderr=subprocess.DEVNULL, text=True
        ).strip()
        if top:
            return Path(top)
    except Exception:
        pass
    # Walk upwards looking for .git
    cur = start
    while True:
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback
    return start

def main():
    p = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and save_to_disk for offline use."
    )
    p.add_argument("dataset", nargs="?", default="glue",
                   help='Dataset name on the Hub (e.g., "glue", "imdb", "ag_news").')
    p.add_argument("config", nargs="?", default=None,
                   help='Optional dataset config (e.g., "mrpc" for GLUE).')
    p.add_argument("out_dir", nargs="?", default=None,
                   help="Output folder. Default: <repo-root>/datasets/<dataset>[-<config>]")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite output folder if it exists.")
    args = p.parse_args()

    repo_root = find_repo_root()
    dir_name = args.dataset if not args.config else f"{args.dataset}-{args.config}"
    out = Path(args.out_dir) if args.out_dir else (repo_root / "datasets" / dir_name)

    if out.exists():
        if not args.force:
            print(f"Error: {out} already exists. Use --force to overwrite.", file=sys.stderr)
            sys.exit(1)
        # Remove existing output if force
        import shutil
        shutil.rmtree(out)

    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.dataset}" + (f"/{args.config}" if args.config else "") + " …")
    ds = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)

    print(f"Saving to {out} …")
    ds.save_to_disk(str(out))
    print("Done.")

if __name__ == "__main__":
    main()
