#!/usr/bin/env python3
"""
Run a sweep of DDPM experiments (multiple configs) and save each run to runs/<run_name>.

Example:
  python tools/run_sweep.py --sweep sweeps/sweep_example.json --epochs 30 --fid-every 1
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class SweepRun:
    name: str
    args: Dict[str, Any]


def _load_sweep(path: Path) -> List[SweepRun]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "runs" not in data:
        raise ValueError("Sweep JSON must be an object with a top-level 'runs' list.")
    runs = data["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("'runs' must be a non-empty list.")
    out: List[SweepRun] = []
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ValueError(f"runs[{i}] must be an object.")
        name = str(r.get("name", "")).strip()
        if not name:
            raise ValueError(f"runs[{i}] missing non-empty 'name'.")
        args = r.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"runs[{i}].args must be an object.")
        out.append(SweepRun(name=name, args=dict(args)))
    return out


def _to_cli_args(k: str, v: Any) -> List[str]:
    """
    Converts {"unet_chs": "64,128"} -> ["--unet-chs", "64,128"]
             {"unet_use_attn": True} -> ["--unet-use-attn"]
             {"unet_use_attn": False} -> ["--unet-no-attn"]
    """
    flag = f"--{k.replace('_', '-')}"
    if isinstance(v, bool):
        if k in {"unet_use_attn"}:
            return ["--unet-use-attn"] if v else ["--unet-no-attn"]
        return [flag] if v else []
    if v is None:
        return []
    return [flag, str(v)]


def _build_cmd(
    run: SweepRun,
    ddpm_py: Path,
    *,
    run_dir: str,
    epochs: int,
    dataset: str,
    max_batches: int,
    max_samples: int,
    seed: int,
    vis_every: int,
    vis_batch_size: int,
    vis_grid_nrow: int,
    vis_fixed_noise: bool,
    fid_every: int,
    fid_num_samples: int,
    fid_batch_size: int,
    fid_device: str,
    fid_use_ema: bool,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(ddpm_py),
        "--mode",
        "train",
        "--dataset",
        dataset,
        "--epochs",
        str(epochs),
        "--max-batches",
        str(max_batches),
        "--max-samples",
        str(max_samples),
        "--run-dir",
        run_dir,
        "--run-name",
        run.name,
        "--seed",
        str(seed),
        "--vis-every",
        str(vis_every),
        "--vis-batch-size",
        str(vis_batch_size),
        "--vis-grid-nrow",
        str(vis_grid_nrow),
        "--fid-every",
        str(fid_every),
        "--fid-num-samples",
        str(fid_num_samples),
        "--fid-batch-size",
        str(fid_batch_size),
        "--fid-device",
        fid_device,
    ]
    if vis_fixed_noise:
        cmd.append("--vis-fixed-noise")
    if fid_use_ema:
        cmd.append("--fid-use-ema")

    for k, v in run.args.items():
        cmd.extend(_to_cli_args(k, v))
    return cmd


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=str, required=True, help="Path to sweep JSON file")
    ap.add_argument("--ddpm", type=str, default="ddpm.py", help="Path to ddpm.py")
    ap.add_argument("--run-dir", type=str, default="runs", help="Base output directory for all runs")

    # Training protocol defaults (fixed across runs for fairness).
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "fake"])
    ap.add_argument("--max-batches", type=int, default=0, help="<=0 means full epoch")
    ap.add_argument("--max-samples", type=int, default=0, help="<=0 means full dataset")
    ap.add_argument("--seed", type=int, default=0)

    # Visualization protocol.
    ap.add_argument("--vis-every", type=int, default=1)
    ap.add_argument("--vis-batch-size", type=int, default=64)
    ap.add_argument("--vis-grid-nrow", type=int, default=8)
    ap.add_argument(
        "--vis-fixed-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fixed noise for per-epoch visualization (recommended for comparisons)",
    )

    # FID protocol.
    ap.add_argument("--fid-every", type=int, default=1)
    ap.add_argument("--fid-num-samples", type=int, default=1000)
    ap.add_argument("--fid-batch-size", type=int, default=64)
    ap.add_argument("--fid-device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--fid-use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EMA weights for FID sampling (recommended)",
    )

    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
    args = ap.parse_args(list(argv))

    sweep_path = Path(args.sweep)
    ddpm_py = Path(args.ddpm)
    if not sweep_path.exists():
        raise SystemExit(f"Missing sweep file: {sweep_path}")
    if not ddpm_py.exists():
        raise SystemExit(f"Missing ddpm file: {ddpm_py}")

    runs = _load_sweep(sweep_path)
    for i, run in enumerate(runs, start=1):
        cmd = _build_cmd(
            run,
            ddpm_py,
            run_dir=args.run_dir,
            epochs=args.epochs,
            dataset=args.dataset,
            max_batches=args.max_batches,
            max_samples=args.max_samples,
            seed=args.seed,
            vis_every=args.vis_every,
            vis_batch_size=args.vis_batch_size,
            vis_grid_nrow=args.vis_grid_nrow,
            vis_fixed_noise=bool(args.vis_fixed_noise),
            fid_every=args.fid_every,
            fid_num_samples=args.fid_num_samples,
            fid_batch_size=args.fid_batch_size,
            fid_device=args.fid_device,
            fid_use_ema=bool(args.fid_use_ema),
        )
        print(f"[{i}/{len(runs)}] run={run.name}")
        print("  " + " ".join(shlex.quote(x) for x in cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
