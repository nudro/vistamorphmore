# VistaMorphMore

Heterogeneous adaptive **visible–thermal (EO/IR)** registration under mild to severe geometric warps.

Builds on:

- [Vista-Morph](https://arxiv.org/abs/2306.06505) ([nudro/VistaMorph](https://github.com/nudro/VistaMorph))
- [When Visible-to-Thermal Facial GAN Beats Conditional Diffusion](https://arxiv.org/abs/2302.09395)

## Diffusion_B

**Latent diffusion:** encode EO → denoise in latent space (ε-UNet) → decode a visible proxy from IR, then a ViT–GNN affine STN registers IR to EO.

**SLIC / RAG:** SLIC superpixels form a RAG; node features are pooled and rasterized to a latent-resolution map that conditions the ε-net. Registration GNN defaults to **DiffSLIC** (`--slic_backend diff`); DDPM graph-cond uses **skimage** SLIC. Pass `--slic_backend skimage` to avoid DiffSLIC.

Lean package: only symbols needed by this trainer (no full CUT / SLIC_GAN trees).

| Module | Role |
|--------|------|
| `paired_dataset.py` | EO/IR pair loaders |
| `weight_init.py` | `weights_init_normal` |
| `slic_features.py` | luma/Sobel helpers + DiffSLIC label map |
| `diffslic_upstream/` | vendored DiffSLIC (MIT) |

## Run

```bash
# smoke (requires a tier with train/)
python train.py --smoke --data_root /path/to/tier --gpu_num 0

# train
DATA_ROOT=/path/to/tier bash run_train.sh

# all_pairs-style curriculum (100 ep, reg@50) when Data/all_pairs is available
bash train_all_pairs.sh
```

Defaults: 256², `ddpm_T=500`, `ddim_steps=32`, `slic_backend=diff`, phase-1 then phase-2 at `--reg_start_epoch` (100 CLI / 50 in `train_all_pairs.sh`).

Sample FLIR strips: `assets/`.
