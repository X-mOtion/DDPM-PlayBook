# DDPM-PlayBook

This repository provides a bilingual playbook for building and tuning diffusion models based on six core papers.

## Contents

- `Books/Diffusion_6Papers_FirstPrinciples_Playbook_zh.md`: Chinese version (source markdown)
- `Books/Diffusion_6Papers_FirstPrinciples_Playbook_zh.pdf`: Chinese version (PDF)
- `Books/Diffusion_6Papers_FirstPrinciples_Playbook_en.md`: English version (source markdown)
- `Books/Diffusion_6Papers_FirstPrinciples_Playbook_en.pdf`: English version (PDF)
- `Books/text/`: extracted page-aligned text used for citation/page alignment

## Scope

- Focuses on reproducible design and tuning workflows for diffusion models.
- Organizes guidance into data processing, model design, objectives, optimization, sampling, and evaluation.
- Uses a structured entry format for implementation and review.

## If this helps

If this repository is useful to you, please leave a star on GitHub.

## Experiments (DDPM demo)

`ddpm.py` supports per-epoch visualization and (CIFAR-10) FID tracking. Each run is stored in `runs/<run_name>/`.

Sweep multiple variants (30 epochs each) and compare best FID:

```bash
python tools/run_sweep.py --sweep sweeps/sweep_example.json --epochs 30 --fid-every 1 --fid-num-samples 1000
python tools/summarize_sweep.py
```
