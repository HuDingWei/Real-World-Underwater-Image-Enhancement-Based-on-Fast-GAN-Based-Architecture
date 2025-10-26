## Overview
This repository contains the **core PyTorch modules** for a research project on **underwater image enhancement**.  
The implementation uses a **UNet-style generator** with **channel-wise feature enhancement**, modular convolution blocks, and skip connections.

> **Note:**  
> - Transformer modules have been removed; the code is fully functional with convolutional blocks.  
> - Training datasets and model weights are **not included** to protect research and potential commercial value.  
> - The repository is intended for **architecture review and code inspection**.

---

## Repository Structure
- `net.py` – Main generator network (UNet-style with enhancement formulas)  
- `block.py` – Core modules and building blocks:
  - Encoder/decoder convolution blocks (`conv_block`, `up_conv`)  
  - Normalization layers (`PixelwiseNorm`, `MinibatchStdDev`)  
  - Feature map conversions (`from_rgb`, `to_rgb`)  

---

## Features
- Modular **encoder/decoder blocks**  
- **Skip connections** for feature fusion  
- Pixel-wise normalization and minibatch standard deviation layers  
- Multi-channel to RGB conversion for discriminator input  
- Fully implemented **forward pass**, ready for integration  
