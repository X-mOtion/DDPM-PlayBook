#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="runs", help="Base runs directory")
    ap.add_argument("--out", type=str, default="runs/best_fid.csv", help="Output CSV path")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        cfg_path = run / "config.json"
        best_path = run / "best_fid.json"
        if not cfg_path.exists() or not best_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        best = json.loads(best_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": run.name,
                "best_fid": best.get("best_fid"),
                "best_epoch": best.get("best_epoch"),
                "T": cfg.get("T"),
                "schedule_type": cfg.get("schedule_type"),
                "lr": cfg.get("lr"),
                "batch_size": cfg.get("batch_size"),
                "unet_chs": ",".join(str(x) for x in cfg.get("unet_chs", [])),
                "unet_num_blocks": cfg.get("unet_num_blocks"),
                "unet_use_attn": cfg.get("unet_use_attn"),
                "unet_attn_heads": cfg.get("unet_attn_heads"),
            }
        )

    # Sort by best_fid ascending (None last)
    def key(r):
        v = r.get("best_fid")
        return (v is None, v if v is not None else 0.0)

    rows.sort(key=key)

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
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(h) is None else str(r.get(h)) for h in header) + "\n")

    print(f"Wrote: {out_path} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

