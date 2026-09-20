#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def pair_key(path: Path) -> str:
    return path.stem.replace("原", "").replace("洇", "")


def collect(split: Path) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    origins = list(split.rglob("origin/*.png"))
    inks = list(split.rglob("ink/*.png"))

    def mapping(paths: list[Path]) -> dict[tuple[str, str], Path]:
        result = {}
        for path in paths:
            group = str(path.parent.parent.relative_to(split))
            key = (group, pair_key(path))
            if key in result:
                raise ValueError(f"duplicate pair key {key}: {result[key]} and {path}")
            result[key] = path.resolve()
        return result

    origin_map, ink_map = mapping(origins), mapping(inks)
    common = sorted(origin_map.keys() & ink_map.keys())
    pairs = []
    for key in common:
        hq, lq = origin_map[key], ink_map[key]
        with Image.open(hq) as hq_image, Image.open(lq) as lq_image:
            if hq_image.size != lq_image.size:
                raise ValueError(f"dimension mismatch: {hq} {hq_image.size}, {lq} {lq_image.size}")
        pairs.append((lq, hq))
    only_hq = [origin_map[key] for key in sorted(origin_map.keys() - ink_map.keys())]
    only_lq = [ink_map[key] for key in sorted(ink_map.keys() - origin_map.keys())]
    return pairs, only_hq, only_lq


def write_manifest(pairs: list[tuple[Path, Path]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("lq", "hq"))
        writer.writerows(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", default="manifests", type=Path)
    args = parser.parse_args()
    splits = {"train": args.root / "train", "test": args.root / "test"}
    unmatched_lines = []
    for name, split in splits.items():
        pairs, only_hq, only_lq = collect(split)
        write_manifest(pairs, args.output / f"ink_bleed_{name}.csv")
        print(f"{name}: {len(pairs)} pairs, HQ-only={len(only_hq)}, LQ-only={len(only_lq)}")
        unmatched_lines.extend(f"{name},hq_only,{path}" for path in only_hq)
        unmatched_lines.extend(f"{name},lq_only,{path}" for path in only_lq)
    (args.output / "ink_bleed_unmatched.txt").write_text(
        "\n".join(unmatched_lines) + ("\n" if unmatched_lines else ""), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

