# 🐾 PAWS: Perception-aligned, Warp-guided Video Super-resolution

**Status:** 🚧 In active development

---

## Project Overview

PAWS is a modular video super-resolution framework designed for both research and deployment. It supports arbitrary power-of-two upscaling and is architected around a configuration system that specifies backbone, discriminator, and optical flow modules. The framework accommodates both distortion-based (PSNR) and perceptual (GAN-based) training, and provides support for custom degradation pipelines for both synthetic and real-world data. Mixed-precision (AMP) training is natively supported for efficient resource usage. Code and models are structured for reproducibility and extensibility, with full experiment tracking and modular integration of new components.


## 💡 Key Features

- **Flexible, modular design:** Modular config-driven pipeline—swap or extend generators, discriminators, and optical flow models with a single config change.
- **Efficient:** Designed for practical deployment on consumer-grade hardware.
- **Custom Dataset Pipelines:** Supports both real-world and synthetic degradations; easy to add new data transforms.
- **Mixed-precision & AMP:** Native support for efficient training and inference on modern GPUs.
- **Powerful logging:** Integrated TensorBoard scalars, images, and GPU memory usage for full experiment tracking.
- **Open-source friendly:** Clear repo structure, readable code, and plug-and-play utilities for research and production.


## 📦 Installation and Setup

Python 3.12 is recommended. Install the PyTorch stack before the rest of the dependencies to avoid accidentally installing CPU-only builds, especially on Windows.

1. Create and activate a Python 3.12 environment:

```bash
conda create -n paws python=3.12
conda activate paws
```

2. Install PyTorch, TorchVision, and TorchAudio using the official [PyTorch Start Locally](https://pytorch.org/get-started/locally/) selector, or use the CUDA 12.6 example below.

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

If you need a different CUDA target, use the matching PyTorch index URL.

3. Install the remaining requirements, then install the pinned COVER dependency with `--no-deps`:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps "cover @ git+https://github.com/taco-group/COVER.git@7373bee6bc55fdc2b617e40c8c45c8295c5e6467"
```

The `--no-deps` flag prevents pip from trying to satisfy COVER's older `torch~=1.13` dependency constraint after the correct PyTorch build is already installed.


## 🔥 Quick Start: Training

Training is driven by YAML config files under `configs/`. The config defines the model, dataset paths, dataloader settings, losses, checkpoint behavior, and logging name.

For single-process training, pass the config path directly to `train.py`:

```bash
python train.py configs/PAWS_RealBasicVSRPP_HAT_Stage1_PSNR_REDSx4.yaml
```

For distributed data parallel training, launch `train_ddp.py` with `torchrun`. Set `--nproc_per_node` to the number of GPU processes to run, typically one process per GPU:

```bash
torchrun --nproc_per_node=2 train_ddp.py configs/PAWS_RealBasicVSRPP_HAT_Stage1_PSNR_REDSx4.yaml
```

If you need to select specific GPUs, set `CUDA_VISIBLE_DEVICES` before launching. If the default distributed port is already in use, pass a different `--master_port` value to `torchrun`.

Prepare paired low-resolution and high-resolution frame folders that match the paths in `dataset.train` and `dataset.val`:

```text
data/REDS/
|-- lr/
|   |-- train/
|   |   |-- <sequence_id>/
|   |   |   |-- <frame>.png
|   |-- val/
|   |   |-- <sequence_id>/
|   |   |   |-- <frame>.png
|-- hr/
|   |-- train/
|   |   |-- <sequence_id>/
|   |   |   |-- <frame>.png
|   |-- val/
|   |   |-- <sequence_id>/
|   |   |   |-- <frame>.png
```

Each LR sequence folder should have a matching HR sequence folder with frames sorted by filename. HR frame dimensions must equal the LR dimensions multiplied by `model.scale_factor`.

Checkpoints are written to `training.save_checkpoint_dir`, final weights are written to `training.final_model_dir`, and TensorBoard logs use `logging.name` under `logs/`.


## 🧪 Quick Start: Testing

Testing uses `test.py` with the same config file used to define the model and test paths. Pass both the config path and the trained generator checkpoint:

```bash
python test.py configs/PAWS_RealBasicVSRPP_HAT_Stage1_PSNR_REDSx4.yaml outputs/PAWS_RealBasicVSRPP_HAT_Stage1_PSNR_REDSx4_G_EMA.pth
```

Use the `_G.pth` checkpoint instead of `_G_EMA.pth` if you want to test the non-EMA generator.

Common optional arguments include `--tile_t`, `--tile_h`, and `--tile_w` for tiled inference; `--pad_t`, `--pad_h`, and `--pad_w` for tile overlap; `--precision {fp32,bf16,fp16}` for inference precision; and `--compile_mode {none,default,reduce-overhead,max-autotune}` for optional `torch.compile` support.

Example tiled inference command:

```bash
python test.py configs/PAWS_RealBasicVSRPP_HAT_Stage1_PSNR_REDSx4.yaml outputs/PAWS_RealBasicVSRPP_HAT_Stage1_PSNR_REDSx4_G_EMA.pth --tile_t 0 --tile_h 256 --tile_w 256 --pad_h 20 --pad_w 20
```

Prepare test input frames under `dataset.test.lr_dir`. Super-resolved frames are saved under matching sequence folders in `dataset.test.hr_dir`:

```text
data/REDS/
|-- lr/
|   |-- test/
|   |   |-- <sequence_id>/
|   |   |   |-- <frame>.png
|-- hr/
|   |-- test/
|   |   |-- <sequence_id>/
|   |   |   |-- 00000000.png
```

Input frames are sorted by filename before inference. Output frames are written as PNG files named `00000000.png`, `00000001.png`, and so on.


## 🚧 Roadmap

- ✅ Modular PyTorch training pipeline
- ✅ Advanced dataset/dataloader (real + synthetic)
- ✅ Custom degradation pipeline
- ✅ Full model config registry (generator/discriminator/flow)
- ✅ Inference & demo scripts
- 🔲 Model weights and sample outputs
- 🔲 Web/desktop demo app
- 🔲 Paper/report


## ✨ Updates

- **June 1, 2026:** Expanded the PAWS training, inference, and evaluation workflow. Added REDS x4 configs for RealBasicVSR++, SSIM loss support, tiled and precision-aware testing with optional `torch.compile`, full-reference/no-reference/Ewarp benchmarking, COVER scoring, checkpoint search, portable model publishing, pinned requirements, and torchrun-compatible DDP launch handling.
- **January 23, 2026:** Added RealBasicVSR++ architecture with modular cleaning backends (RRDB, SwinIR, HAT) and dynamic refinement. Major refactor of model registry, training compatibility, and checkpointing with correct mid-epoch resume support.
- **June 23, 2025:** Added inference script for test-time video super-resolution
- **May 31, 2025:** Project repository initialized - dataset, training, modular config, and model registry


## 🤖 Model Zoo

| Backbone  | Model  | Params  | Config  | Download  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| BasicVSR++  | PAWS_BasicVSRPP_PSNR_REDSx4  | 7.7M  | -  | Coming Soon  |
| BasicVSR++  |  PAWS_BasicVSRPP_GAN_REDSx4  | 7.7M  | -  | Coming Soon  |
| BasicVSR++  | PAWS_BasicVSRPP_Real_PSNR_REDSx4  | 7.7M  | -  | Coming Soon  |
| BasicVSR++  | PAWS_BasicVSRPP_Real_GAN_REDSx4  | 7.7M  | -  | Coming Soon  |


## ⚖️ License and Acknowledgement

This project is released under the Apache 2.0 license.
See the [NOTICE](NOTICE.md) file for acknowledgements and third-party code attributions.
