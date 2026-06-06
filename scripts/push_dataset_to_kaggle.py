"""Publish processed parquet shards as a versioned Kaggle Dataset.

Two backends:
  - "mcp": uses the Kaggle MCP server (preferred when running inside an
    MCP-enabled session). Tool names will be filled in once the MCP is wired.
  - "cli": uses the `kaggle` CLI (default; works locally and on Kaggle itself).

Usage:
  python scripts/push_dataset_to_kaggle.py \
      --dir data/processed \
      --slug nipung2010/lichess-blunder-shards-v1 \
      --title "Lichess blunder shards v1" \
      --backend cli
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def write_metadata(directory: Path, slug: str, title: str) -> None:
    owner, name = slug.split("/", 1)
    metadata = {
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
        "resources": [
            {"path": p.name, "description": p.name}
            for p in sorted(directory.glob("*"))
            if p.is_file() and p.name != "dataset-metadata.json"
        ],
    }
    (directory / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))


def push_cli(directory: Path, slug: str, title: str, version_notes: str) -> None:
    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found. Install with `pip install kaggle` and set KAGGLE_USERNAME/KAGGLE_KEY.")
    write_metadata(directory, slug, title)
    meta = directory / "dataset-metadata.json"
    if not meta.exists():
        sys.exit(f"Missing {meta}")
    cmd = ["kaggle", "datasets", "version", "-p", str(directory), "-m", version_notes, "--dir-mode", "zip"]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("Version push failed; trying initial create.", file=sys.stderr)
        cmd = ["kaggle", "datasets", "create", "-p", str(directory), "--dir-mode", "zip"]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)


def push_mcp(directory: Path, slug: str, title: str, version_notes: str) -> None:
    # Will be implemented once the Kaggle MCP server's tool schema lands in this
    # session. The signature is intentionally identical to push_cli so callers
    # can flip backends without changes.
    raise NotImplementedError(
        "MCP backend not wired yet. Use --backend cli until kaggle_* tools appear."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True, help="directory containing parquet shards")
    p.add_argument("--slug", required=True, help="kaggle dataset slug (owner/name)")
    p.add_argument("--title", required=True)
    p.add_argument("--notes", default="automated version push")
    p.add_argument("--backend", choices=("cli", "mcp"), default="cli")
    args = p.parse_args()

    if not args.dir.is_dir():
        sys.exit(f"Not a directory: {args.dir}")
    if args.backend == "cli":
        push_cli(args.dir, args.slug, args.title, args.notes)
    else:
        push_mcp(args.dir, args.slug, args.title, args.notes)


if __name__ == "__main__":
    main()
