import argparse
import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, utils as vutils


class BetaScheduler:
    """Generate beta/alpha schedules for diffusion."""

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        self.T = T
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        elif schedule_type == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            alphas_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = betas.clamp(1e-4, 0.999).float()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / max(half_dim - 1, 1)
        )
        angles = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class UNetBlock(nn.Module):
    """Conv + time conditioning + residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.shortcut(x)


class SelfAttention2d(nn.Module):
    """Spatial self-attention over (H*W) tokens."""

    def __init__(self, ch: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        # Choose a head count that divides channels to avoid shape errors.
        h = int(num_heads)
        h = max(1, min(h, ch))
        while ch % h != 0 and h > 1:
            h -= 1
        self.num_heads = h
        self.head_dim = ch // self.num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.GroupNorm(num_groups=min(num_groups, ch), num_channels=ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor, _t_emb: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        qkv = self.qkv(y)
        q, k, v = qkv.chunk(3, dim=1)

        # (B, heads, head_dim, HW)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        # Attention: (B, heads, HW, HW)
        attn = torch.einsum("bhdi,bhdj->bhij", q * self.scale, k)
        attn = attn.softmax(dim=-1)

        # Output: (B, heads, head_dim, HW) -> (B, C, H, W)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).contiguous()
        out = out.view(b, c, h, w)
        out = self.proj(out)
        return x + out


class UNet(nn.Module):
    """A compact UNet for CIFAR-10 denoising."""

    def __init__(
        self,
        in_ch: int = 3,
        chs: Sequence[int] = (64, 128, 256),
        time_dim: int = 128,
        num_blocks: int = 2,
        num_groups: int = 32,
        attn_heads: int = 4,
        use_attn: bool = True,
    ):
        super().__init__()
        if len(chs) < 1:
            raise ValueError("chs must contain at least one channel width")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.time_embed = TimeEmbedding(time_dim)
        self.in_ch = in_ch
        self.chs = tuple(int(c) for c in chs)
        self.num_blocks = int(num_blocks)
        self.num_groups = int(num_groups)
        self.attn_heads = int(attn_heads)
        self.use_attn = bool(use_attn)

        # Stage-wise structure keeps skip connections well-defined even when num_blocks/chs change.
        self.encoder_stages = nn.ModuleList()
        prev = in_ch
        for stage_ch in self.chs:
            # Per stage: 2 residual blocks + 1 attention block.
            blocks: list[nn.Module] = [
                UNetBlock(prev, stage_ch, time_dim, num_groups=self.num_groups),
            ]
            for _ in range(max(0, self.num_blocks - 1)):
                blocks.append(UNetBlock(stage_ch, stage_ch, time_dim, num_groups=self.num_groups))
            if self.use_attn:
                blocks.append(SelfAttention2d(stage_ch, num_heads=self.attn_heads, num_groups=self.num_groups))
            self.encoder_stages.append(nn.ModuleList(blocks))
            prev = stage_ch

        self.decoder_stages = nn.ModuleList()
        # Only (len(chs)-1) upsampling transitions.
        for stage_idx in reversed(range(len(self.chs) - 1)):
            target_ch = self.chs[stage_idx]
            blocks = [UNetBlock(prev, target_ch, time_dim, num_groups=self.num_groups)]
            for _ in range(max(0, self.num_blocks - 1)):
                blocks.append(UNetBlock(target_ch, target_ch, time_dim, num_groups=self.num_groups))
            if self.use_attn:
                blocks.append(SelfAttention2d(target_ch, num_heads=self.attn_heads, num_groups=self.num_groups))
            self.decoder_stages.append(nn.ModuleList(blocks))
            prev = target_ch

        self.out_block = UNetBlock(prev, in_ch, time_dim, num_groups=self.num_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        skips: list[torch.Tensor] = []
        for stage_idx, blocks in enumerate(self.encoder_stages):
            for block in blocks:
                x = block(x, t_emb)
            skips.append(x)
            # Do not downsample after the last encoder stage.
            if stage_idx != len(self.encoder_stages) - 1:
                x = F.avg_pool2d(x, 2)

        # Decode: each stage upsamples once, aligns channels, then adds the skip of that stage.
        # skips[-1] corresponds to the deepest stage and is already in x; start from the next one.
        for blocks in self.decoder_stages:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = skips.pop(-2)
            for bi, block in enumerate(blocks):
                x = block(x, t_emb)
                if bi == 0:
                    x = x + skip

        x = self.out_block(x, t_emb)
        return x


class DDPM(nn.Module):
    """DDPM wrapper with training loss and ancestral sampling."""

    def __init__(
        self,
        T: int = 1000,
        schedule_type: str = "linear",
        unet_chs: Sequence[int] = (64, 128, 256),
        unet_time_dim: int = 128,
        unet_num_blocks: int = 2,
        unet_num_groups: int = 32,
        unet_attn_heads: int = 4,
        unet_use_attn: bool = True,
    ):
        super().__init__()
        self.T = T
        self.schedule_type = schedule_type
        self.model = UNet(
            chs=unet_chs,
            time_dim=unet_time_dim,
            num_blocks=unet_num_blocks,
            num_groups=unet_num_groups,
            attn_heads=unet_attn_heads,
            use_attn=unet_use_attn,
        )

        scheduler = BetaScheduler(T=T, schedule_type=schedule_type)
        self.register_buffer("betas", scheduler.betas)
        self.register_buffer("alphas", scheduler.alphas)
        self.register_buffer("alpha_bars", scheduler.alpha_bars)
        alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=scheduler.alpha_bars.dtype), scheduler.alpha_bars[:-1]], dim=0
        )
        posterior_variance = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        posterior_variance = posterior_variance.clamp(min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        bsz = x0.shape[0]
        t = torch.randint(0, self.T, (bsz,), device=x0.device)
        eps = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
        eps_pred = self.model(xt, t)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        img_size=(32, 32),
        denoise_model: nn.Module | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if noise is None:
            x = torch.randn(batch_size, 3, *img_size, device=device)
        else:
            x = noise.to(device)
        denoise_model = self.model if denoise_model is None else denoise_model

        for t_int in reversed(range(self.T)):
            t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
            alpha_t = self.alphas[t_int]
            alpha_bar_t = self.alpha_bars[t_int]

            eps_pred = denoise_model(x, t)
            coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mu = (x - coef * eps_pred) / torch.sqrt(alpha_t)

            if t_int > 0:
                z = torch.randn_like(x)
                sigma = torch.sqrt(self.posterior_variance[t_int])
                x = mu + sigma * z
            else:
                x = mu

        return x.clamp(-1.0, 1.0)


@dataclass
class TrainConfig:
    mode: str
    dataset: str
    epochs: int
    batch_size: int
    lr: float
    max_batches: int
    max_samples: int
    T: int
    schedule_type: str
    sample_batch_size: int
    sample_out: str
    num_workers: int
    log_every: int
    ema_decay: float
    ckpt_in: str
    ckpt_out: str
    resume: bool
    unet_chs: tuple[int, ...]
    unet_time_dim: int
    unet_num_blocks: int
    unet_num_groups: int
    unet_attn_heads: int
    unet_use_attn: bool
    run_dir: str
    run_name: str
    seed: int
    vis_every: int
    vis_batch_size: int
    vis_grid_nrow: int
    vis_fixed_noise: bool
    fid_every: int
    fid_num_samples: int
    fid_batch_size: int
    fid_device: str
    fid_use_ema: bool


def build_dataloader(cfg: TrainConfig) -> DataLoader:
    if cfg.dataset == "fake":
        n = cfg.max_samples if cfg.max_samples > 0 else 2048
        x = torch.randn(n, 3, 32, 32).clamp(-1.0, 1.0)
        y = torch.zeros(n, dtype=torch.long)
        dataset = TensorDataset(x, y)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    if cfg.max_samples > 0:
        dataset = Subset(dataset, list(range(min(cfg.max_samples, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def _as_jsonable_cfg(cfg: TrainConfig) -> dict:
    d = cfg.__dict__.copy()
    # Normalize tuples for JSON.
    d["unet_chs"] = list(cfg.unet_chs)
    return d


def _prepare_run_dir(cfg: TrainConfig) -> Path:
    base = Path(cfg.run_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = cfg.run_name.strip() if cfg.run_name.strip() else f"run_{ts}"
    run_dir = base / name
    # Avoid collisions.
    if run_dir.exists():
        run_dir = base / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.json").write_text(json.dumps(_as_jsonable_cfg(cfg), indent=2), encoding="utf-8")
    return run_dir


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_image_dir(samples_01: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save individual PNGs for torch-fidelity.
    for i in range(samples_01.shape[0]):
        vutils.save_image(samples_01[i], str(out_dir / f"{i:06d}.png"))


@torch.no_grad()
def _compute_fid_cifar10(
    ddpm: "DDPM",
    denoise_model: nn.Module,
    cfg: TrainConfig,
    run_dir: Path,
    epoch: int,
    device: str,
) -> float:
    """
    Compute FID against CIFAR-10 train split using torch-fidelity.
    We generate cfg.fid_num_samples images into a directory and compare to CIFAR-10 dataset.
    """
    import torch_fidelity

    ddpm.eval()
    denoise_model.eval()

    gen_dir = run_dir / "fid" / f"epoch_{epoch:04d}" / "gen"
    if gen_dir.exists():
        # If rerun, avoid mixing old/new.
        for p in gen_dir.glob("*.png"):
            p.unlink()
    gen_dir.mkdir(parents=True, exist_ok=True)

    remaining = cfg.fid_num_samples
    idx = 0
    while remaining > 0:
        bs = min(cfg.fid_batch_size, remaining)
        samples = ddpm.sample(batch_size=bs, img_size=(32, 32), denoise_model=denoise_model)
        samples = (samples + 1.0) / 2.0
        # Write batch as individual images.
        for i in range(samples.shape[0]):
            vutils.save_image(samples[i], str(gen_dir / f"{idx:06d}.png"))
            idx += 1
        remaining -= bs

    metrics = torch_fidelity.calculate_metrics(
        input1=str(gen_dir),
        input2="cifar10-train",
        cuda=(cfg.fid_device == "cuda"),
        isc=False,
        kid=False,
        prc=False,
        fid=True,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def save_ckpt(path: str, ddpm: DDPM, ema_model: nn.Module, cfg: TrainConfig) -> None:
    torch.save(
        {
            "model": ddpm.model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "T": ddpm.T,
            "schedule_type": ddpm.schedule_type,
            "unet_chs": ddpm.model.chs if hasattr(ddpm.model, "chs") else None,
            "unet_time_dim": cfg.unet_time_dim,
            "unet_num_blocks": cfg.unet_num_blocks,
            "unet_num_groups": cfg.unet_num_groups,
            "unet_attn_heads": cfg.unet_attn_heads,
            "unet_use_attn": cfg.unet_use_attn,
        },
        path,
    )
    print(f"Saved checkpoint to: {path}", flush=True)


def load_ckpt(path: str, device: str) -> tuple[DDPM, nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    unet_chs = ckpt.get("unet_chs") or (64, 128, 256)
    ddpm = DDPM(
        T=int(ckpt["T"]),
        schedule_type=str(ckpt["schedule_type"]),
        unet_chs=tuple(int(c) for c in unet_chs),
        unet_time_dim=int(ckpt.get("unet_time_dim", 128)),
        unet_num_blocks=int(ckpt.get("unet_num_blocks", 2)),
        unet_num_groups=int(ckpt.get("unet_num_groups", 32)),
        unet_attn_heads=int(ckpt.get("unet_attn_heads", 4)),
        unet_use_attn=bool(ckpt.get("unet_use_attn", True)),
    ).to(device)
    ddpm.model.load_state_dict(ckpt["model"])
    ema_model = copy.deepcopy(ddpm.model).to(device).eval()
    ema_model.load_state_dict(ckpt["ema_model"])
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ddpm, ema_model, ckpt


def run_sampling(ddpm: DDPM, ema_model: nn.Module, cfg: TrainConfig, device: str) -> None:
    ddpm.eval()
    samples = ddpm.sample(batch_size=cfg.sample_batch_size, img_size=(32, 32), denoise_model=ema_model)
    samples = (samples + 1.0) / 2.0
    vutils.save_image(samples, cfg.sample_out, nrow=int(math.sqrt(cfg.sample_batch_size)))
    print(f"Saved samples to: {cfg.sample_out}", flush=True)
    print(
        f"Sample tensor stats => min={samples.min().item():.4f}, "
        f"max={samples.max().item():.4f}, mean={samples.mean().item():.4f}",
        flush=True,
    )


def train_and_sample(cfg: TrainConfig) -> None:
    device = pick_device()
    print(f"Device: {device}", flush=True)

    run_dir = _prepare_run_dir(cfg)
    print(f"Run dir: {run_dir}", flush=True)

    dataloader = build_dataloader(cfg)
    if cfg.resume:
        model, ema_model, ckpt = load_ckpt(cfg.ckpt_in, device)
        print(
            f"Resume from checkpoint => path={cfg.ckpt_in}, T={ckpt['T']}, schedule={ckpt['schedule_type']}",
            flush=True,
        )
    else:
        model = DDPM(
            T=cfg.T,
            schedule_type=cfg.schedule_type,
            unet_chs=cfg.unet_chs,
            unet_time_dim=cfg.unet_time_dim,
            unet_num_blocks=cfg.unet_num_blocks,
            unet_num_groups=cfg.unet_num_groups,
            unet_attn_heads=cfg.unet_attn_heads,
            unet_use_attn=cfg.unet_use_attn,
        ).to(device)
        ema_model = copy.deepcopy(model.model).to(device).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    print(
        f"Train config => dataset={cfg.dataset}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"T={model.T}, max_batches={cfg.max_batches}, max_samples={cfg.max_samples}, resume={cfg.resume}",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Fixed noise for comparable per-epoch visualization.
    vis_noise = None
    if cfg.vis_fixed_noise:
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        vis_noise = torch.randn(cfg.vis_batch_size, 3, 32, 32, generator=g)

    metrics_path = run_dir / "metrics.jsonl"
    best_fid = float("inf")
    best_fid_epoch = -1
    best_ckpt_path = run_dir / "ddpm_best_fid_ckpt.pt"

    model.train()
    for epoch in range(cfg.epochs):
        for batch_idx, (x0, _) in enumerate(dataloader):
            if cfg.max_batches > 0 and batch_idx >= cfg.max_batches:
                break
            x0 = x0.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x0)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.model.parameters()):
                    p_ema.copy_(p_ema * cfg.ema_decay + p * (1.0 - cfg.ema_decay))
            if batch_idx % cfg.log_every == 0:
                print(f"Epoch {epoch:03d} | Batch {batch_idx:04d} | Loss {loss.item():.6f}", flush=True)

        if cfg.vis_every > 0 and ((epoch + 1) % cfg.vis_every == 0):
            model.eval()
            out_path = run_dir / f"samples_epoch_{epoch:04d}.png"
            samples = model.sample(
                batch_size=cfg.vis_batch_size,
                img_size=(32, 32),
                denoise_model=ema_model,
                noise=vis_noise,
            )
            samples = (samples + 1.0) / 2.0
            vutils.save_image(samples, str(out_path), nrow=cfg.vis_grid_nrow)
            print(f"Saved epoch samples to: {out_path}", flush=True)
            model.train()

        if cfg.dataset == "cifar10" and cfg.fid_every > 0 and ((epoch + 1) % cfg.fid_every == 0):
            denoise = ema_model if cfg.fid_use_ema else model.model
            fid = _compute_fid_cifar10(model, denoise, cfg, run_dir, epoch, device=device)
            _append_jsonl(
                metrics_path,
                {
                    "epoch": int(epoch),
                    "fid": fid,
                    "fid_num_samples": int(cfg.fid_num_samples),
                    "fid_use_ema": bool(cfg.fid_use_ema),
                    "T": int(cfg.T),
                    "schedule_type": cfg.schedule_type,
                    "unet_chs": list(cfg.unet_chs),
                    "unet_num_blocks": int(cfg.unet_num_blocks),
                    "unet_use_attn": bool(cfg.unet_use_attn),
                },
            )
            print(f"FID @ epoch {epoch:03d}: {fid:.4f}", flush=True)
            if fid < best_fid:
                best_fid = fid
                best_fid_epoch = epoch
                save_ckpt(str(best_ckpt_path), model, ema_model, cfg)

    ckpt_path = run_dir / Path(cfg.ckpt_out).name
    save_ckpt(str(ckpt_path), model, ema_model, cfg)
    # Final sampling artifact (kept for backwards compatibility).
    final_out = run_dir / Path(cfg.sample_out).name
    cfg2 = copy.deepcopy(cfg)
    cfg2.sample_out = str(final_out)
    run_sampling(model, ema_model, cfg2, device)
    if best_fid_epoch >= 0:
        (run_dir / "best_fid.json").write_text(
            json.dumps(
                {
                    "best_fid": best_fid,
                    "best_epoch": best_fid_epoch,
                    "best_ckpt": str(best_ckpt_path.name),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Best FID: {best_fid:.4f} @ epoch {best_fid_epoch:03d}", flush=True)


def sample_only(cfg: TrainConfig) -> None:
    device = pick_device()
    print(f"Device: {device}", flush=True)
    ddpm, ema_model, ckpt = load_ckpt(cfg.ckpt_in, device)
    print(
        f"Loaded checkpoint => T={ckpt['T']}, schedule={ckpt['schedule_type']}, path={cfg.ckpt_in}",
        flush=True,
    )
    run_sampling(ddpm, ema_model, cfg, device)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="DDPM CIFAR-10 demo.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "fake"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-batches", type=int, default=20, help="<=0 means full epoch")
    parser.add_argument("--max-samples", type=int, default=5000, help="<=0 means full dataset")
    parser.add_argument("--T", type=int, default=200, help="diffusion steps")
    parser.add_argument("--schedule-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--sample-out", type=str, default="ddpm_samples.png")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ckpt-in", type=str, default="ddpm_cifar_ckpt.pt")
    parser.add_argument("--ckpt-out", type=str, default="ddpm_cifar_ckpt.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--unet-chs",
        type=str,
        default="64,128,256",
        help="Comma-separated UNet channel widths per stage, e.g. 64,128,256",
    )
    parser.add_argument("--unet-time-dim", type=int, default=128, help="Time embedding dimension")
    parser.add_argument("--unet-num-blocks", type=int, default=2, help="Residual blocks per stage (attention is added per stage)")
    parser.add_argument("--unet-num-groups", type=int, default=32, help="GroupNorm groups (clamped by channels)")
    parser.add_argument("--unet-attn-heads", type=int, default=4, help="Self-attention heads per stage (auto-adjusted to divide channels)")
    parser.add_argument("--unet-use-attn", action="store_true", help="Enable self-attention blocks in each stage")
    parser.add_argument(
        "--unet-no-attn",
        action="store_true",
        help="Disable self-attention blocks in each stage (overrides --unet-use-attn)",
    )
    parser.add_argument("--run-dir", type=str, default="runs", help="Base directory for experiment outputs")
    parser.add_argument("--run-name", type=str, default="", help="Run name (default: timestamped)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for fixed visualization noise")
    parser.add_argument("--vis-every", type=int, default=1, help="Save visualization every N epochs (<=0 disables)")
    parser.add_argument("--vis-batch-size", type=int, default=64, help="Number of images in per-epoch visualization")
    parser.add_argument("--vis-grid-nrow", type=int, default=8, help="Grid nrow for per-epoch visualization")
    parser.add_argument("--vis-fixed-noise", action="store_true", help="Use fixed noise for per-epoch visualization")
    parser.add_argument("--fid-every", type=int, default=0, help="Compute FID every N epochs (cifar10 only; <=0 disables)")
    parser.add_argument("--fid-num-samples", type=int, default=1000, help="Number of generated images for FID")
    parser.add_argument("--fid-batch-size", type=int, default=64, help="Batch size for FID sample generation")
    parser.add_argument("--fid-device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for FID feature extraction")
    parser.add_argument("--fid-use-ema", action="store_true", help="Use EMA weights for FID sampling (recommended)")
    args = parser.parse_args()
    unet_chs = tuple(int(x.strip()) for x in args.unet_chs.split(",") if x.strip())
    unet_use_attn = bool(args.unet_use_attn)
    if args.unet_no_attn:
        unet_use_attn = False
    return TrainConfig(
        mode=args.mode,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        T=args.T,
        schedule_type=args.schedule_type,
        sample_batch_size=args.sample_batch_size,
        sample_out=args.sample_out,
        num_workers=args.num_workers,
        log_every=args.log_every,
        ema_decay=args.ema_decay,
        ckpt_in=args.ckpt_in,
        ckpt_out=args.ckpt_out,
        resume=args.resume,
        unet_chs=unet_chs,
        unet_time_dim=args.unet_time_dim,
        unet_num_blocks=args.unet_num_blocks,
        unet_num_groups=args.unet_num_groups,
        unet_attn_heads=args.unet_attn_heads,
        unet_use_attn=unet_use_attn,
        run_dir=args.run_dir,
        run_name=args.run_name,
        seed=args.seed,
        vis_every=args.vis_every,
        vis_batch_size=args.vis_batch_size,
        vis_grid_nrow=args.vis_grid_nrow,
        vis_fixed_noise=args.vis_fixed_noise,
        fid_every=args.fid_every,
        fid_num_samples=args.fid_num_samples,
        fid_batch_size=args.fid_batch_size,
        fid_device=args.fid_device,
        fid_use_ema=bool(args.fid_use_ema),
    )


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.mode == "train":
        train_and_sample(cfg)
    else:
        sample_only(cfg)
