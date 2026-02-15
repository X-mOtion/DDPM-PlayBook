#!/usr/bin/env python3
"""
Summarize sweep results by reading runs/*/{config.json,best_fid.json}.

Outputs:
  - runs/best_fid.csv (same as collect_best_fid.py)
  - runs/sweep_report.md (top-k table + grouped stats)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        cfg_path = run / "config.json"
        best_path = run / "best_fid.json"
        if not cfg_path.exists() or not best_path.exists():
            continue
        cfg = _read_json(cfg_path)
        best = _read_json(best_path)
        rows.append(
            {
                "run": run.name,
                "best_fid": best.get("best_fid"),
                "best_epoch": best.get("best_epoch"),
                "schedule_type": cfg.get("schedule_type"),
                "T": cfg.get("T"),
                "lr": cfg.get("lr"),
                "batch_size": cfg.get("batch_size"),
                "unet_chs": ",".join(str(x) for x in cfg.get("unet_chs", [])),
                "unet_num_blocks": cfg.get("unet_num_blocks"),
                "unet_use_attn": cfg.get("unet_use_attn"),
                "unet_attn_heads": cfg.get("unet_attn_heads"),
            }
        )
    rows.sort(key=lambda r: (r["best_fid"] is None, r["best_fid"] if r["best_fid"] is not None else 0.0))
    return rows


def _write_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    header = [
        "run",
        "best_fid",
        "best_epoch",
        "T",
        "schedule_type",
        "lr",
        "batch_size",
        "unet_chs",
        "unet_num_blocks",
        "unet_use_attn",
        "unet_attn_heads",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(h) is None else str(r.get(h)) for h in header) + "\n")


def _group_stats(rows: List[Dict[str, Any]], key: str) -> List[Tuple[str, int, float, float]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        fid = r.get("best_fid")
        if fid is None:
            continue
        buckets[str(r.get(key))].append(float(fid))
    stats = []
    for k, fids in buckets.items():
        stats.append((k, len(fids), sum(fids) / len(fids), min(fids)))
    stats.sort(key=lambda x: x[2])
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="runs")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out-md", type=str, default="runs/sweep_report.md")
    ap.add_argument("--out-csv", type=str, default="runs/best_fid.csv")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs dir: {runs_dir}")

    rows = _scan_runs(runs_dir)
    _write_csv(Path(args.out_csv), rows)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    group_keys = ["unet_use_attn", "unet_num_blocks", "unet_chs", "schedule_type", "T", "lr", "batch_size"]
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Sweep Report\n\n")
        f.write(f"Runs scanned: {len(rows)}\n\n")
        f.write("## Top Runs (by best FID)\n\n")
        f.write("| rank | run | best_fid | best_epoch | unet_chs | blocks | attn | heads | T | schedule |\n")
        f.write("|---:|---|---:|---:|---|---:|---:|---:|---:|---|\n")
        for i, r in enumerate(rows[: max(args.topk, 1)], start=1):
            f.write(
                f"| {i} | {r['run']} | {r.get('best_fid','')} | {r.get('best_epoch','')} | {r.get('unet_chs','')} | "
                f"{r.get('unet_num_blocks','')} | {r.get('unet_use_attn','')} | {r.get('unet_attn_heads','')} | "
                f"{r.get('T','')} | {r.get('schedule_type','')} |\n"
            )
        f.write("\n")

        f.write("## Grouped Stats (mean/min best FID)\n\n")
        for k in group_keys:
            stats = _group_stats(rows, k)
            if not stats:
                continue
            f.write(f"### {k}\n\n")
            f.write("| value | n | mean_best_fid | min_best_fid |\n")
            f.write("|---|---:|---:|---:|\n")
            for v, n, mean_fid, min_fid in stats:
                f.write(f"| {v} | {n} | {mean_fid:.4f} | {min_fid:.4f} |\n")
            f.write("\n")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

