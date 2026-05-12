# Stylus:Repurposing Image Diffusion Models for Training-Free Music Style Transfer on Mel-Spectrograms — ICIP 2026

This document maps each experiment reported in [**"Repurposing Image Diffusion Models for Training-Free Music Style Transfer on Mel-Spectrograms"**](https://arxiv.org/abs/2411.15913) to the scripts that produced it.
- Project page: [The Official Website for Stylus.]()

---

## 0. Installation

### Environment

```bash
conda env create -f environment.yaml
conda activate stylus
```

### External dependency — riffusion-hobby

Several scripts (`main.py`, `main_nophase.py`, `main_interpolation.py`, `get_mel_images.py`, `get_mel_images_all.py`) rely on `util/riffusion_params.py` from the [riffusion-hobby](https://github.com/hmartiro/riffusion-hobby) repository.

Clone it **inside** this directory before running any script:

```bash
# Run from the Stylus/ directory
git clone https://github.com/hmartiro/riffusion-hobby.git
```

Expected layout after cloning:

```
Stylus/
├── main.py
├── riffusion-hobby/
│   └── util/
│       └── riffusion_params.py
└── ...
```

All scripts are designed to be launched from the **parent** directory of `Stylus/` (e.g., `torchrun ./Stylus/main.py ...`), so the path is resolved as `./Stylus/riffusion-hobby`.

---

## 1. Default Stylus Hyperparameters

The Stylus configuration used in every "headline" number (Table 1, qualitative figures) is:

| Symbol | Value | Meaning |
| --- | --- | --- |
| `--gamma` | `0.75` | Query-preservation weight (γ in Eq. 1) |
| `--alpha` | `0.9` | CFG-inspired style guidance scale (Eq. 3) |
| `--temperature` | `1` | Attention temperature for style injection |
| `--ddim_inv_steps` | `50` | DDIM inversion / sampling steps |
| `--save_feat_steps` | `50` | Steps at which K/V features are cached |
| Backbone | `Stable Diffusion v1.5` | Image diffusion prior |
| Phase | Phase-preserving | Content-phase reuse for waveform reconstruction |

---

## 1.1 Required Model Checkpoints

Before running any script, two pretrained diffusion checkpoints must be placed under `models/`:

| Backbone | Expected path | Required files |
| --- | --- | --- |
| Stable Diffusion v1.5 (default) | `models/ldm/stable-diffusion-v1/` | `model.ckpt`, `v1-inference.yaml` |
| Stable Diffusion XL base 1.0 (SDXL ablation) | `models/sgm/stable-diffusion-xl-base-1.0/` | `sd_xl_base_1.0.safetensors`, `sd_xl_base.yaml` |

These paths are wired into the entry-points via `--model_config` / `--ckpt` defaults; if you place checkpoints elsewhere, override those flags on the command line. The interpolation project shares the SD v1.5 checkpoint location at `models/ldm/stable-diffusion-v1/`.

---

## 2. Main Results (Section 4, Table 1)

Quantitative comparison vs **MusicGen** and **MusicTI** and qualitative comparison in **Figure 2**.

### Stylus generation (ours)

| Script | Entry-point | Notes |
| --- | --- | --- |
| `scripts/main.sh` | `main.py` | **Default Stylus run** — γ=0.75, α=0.9, T=1, 50 DDIM steps. Produces the 13,246 stylized samples reported in the paper. |
| `scripts/main_interactive.sh` | `main.py` | Interactive single-pair version of the above (debug / pilot study) |

### Baselines reproduced in-house

Auxiliary `run_check_inference_time.py` reproduces **MusicGen** outputs and measures inference time / GPU memory used for the *Efficiency* rows in Table 1. The MusicTI baseline was retrained per the authors' protocol (no script in this repo).

### Metric calculation

| Script | Metric | Used for |
| --- | --- | --- |
| `metrics/metric_calculation_CP.py` | Content preservation (CLAP) | Table 1 — *Content* |
| `metrics/metric_calculation_SF.py` | Style fit (CLAP; M_Style + S_Style) | Table 1 — *M_Style, S_Style* |
| `metrics/metric_calculation_FAD.py` | FAD-VGG / FAD-CLAP | Reference computation (commented FAD rows in `main.tex`) |
| `metrics/run_check_inference_time.py` | Inference time + memory | Table 1 — *Efficiency* rows |

---

## 3. Multi-style Interpolation (Section 4.1, Table 3)

All runs call `main_interpolation.py`, which adds two style inputs (`--sty1`, `--sty2`) and a mixing scalar `--mix_beta` (β in the paper).

### 3.1 Headline interpolation sweep (Table 3)

| Script | β | Style A weight | Style B weight |
| --- | --- | --- | --- |
| `default_beta0.1.sh` | 0.1 | 0.9 | 0.1 |
| `default_beta0.3.sh` | 0.3 | 0.7 | 0.3 |
| `default_beta0.5.sh` | 0.5 | 0.5 | 0.5 |
| `default_beta0.7.sh` | 0.7 | 0.3 | 0.7 |
| `default_beta0.9.sh` | 0.9 | 0.1 | 0.9 |

All five use γ=0.75, α=0.9, `--without_init_adain`, 50 DDIM steps. Reference style pairs are drawn from `{accordion, harp, jaw, empty, heartbeat, cornet, erhu}` over the content subset `{color, piano, relax, relieve, sad_violin, twinkle, village, violin}`.

### 3.2 Interpolation baseline (no CFG)

`scripts/default_beta0.5_only_StyleID.sh` — `α=1` (pure key/value swap) at β=0.5, used as a reference point for the CFG-interpolation comparison.

### 3.3 Metrics

| Script | Metric | Purpose |
| --- | --- | --- |
| `metrics/metric_calculation_SF_interpolation.py` | Style fit vs Style A and Style B | Table 3 columns |
| `metrics/metric_calculation_CP_interpolation.py` | Content preservation | Sanity check on structural integrity under interpolation |

---

## 4. Supporting Utilities

These are not standalone experiments but are run as preprocessing / postprocessing around the scripts above.

### Audio ↔ Mel-spectrogram conversion

| Script | Role |
| --- | --- |
| `get_mel_images.py`, `get_mel_images_all.py` | STFT → Mel-spectrogram image preparation for the diffusion pipeline |
| `change_image_scale.py` | Image-domain rescaling helper |

### Audio resampling for fair evaluation

| Script | Purpose |
| --- | --- |
| `resample_orig_audio.py` | Resample original MusicTI clips to a common rate |
| `resample_benchmark_audio.py` | Resample MusicGen / MusicTI baseline outputs |
| `resample_Must_stylized_audio.py` | Resample Stylus outputs |

### Data sanity

| Script | Purpose |
| --- | --- |
| `search_NG_combination.py` | Identify content–style pairs to exclude (stored under `nonexisting_combination/`) |
