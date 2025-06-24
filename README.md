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
