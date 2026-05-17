# Modified SRGAN (MSRGAN) — Image Super-Resolution

A PyTorch implementation of a modified Super-Resolution Generative Adversarial Network (SRGAN), referred to throughout the code as **MSRGAN**. The model enhances the quality of low-resolution images using a GAN framework with Residual in Residual Dense Blocks (RRDB) and perceptual loss derived from a pretrained ResNet-18.

---

## Overview

This notebook implements an end-to-end pipeline for image super-resolution on a furniture image dataset. It covers dataset preparation, model definition, training with early stopping, evaluation using PSNR and SSIM, and visual inspection of results.

---

## Architecture

### Generator
The generator uses a modified ESRGAN-style backbone:

- **Input conv layer** → PReLU activation
- **Second conv layer** → LeakyReLU activation
- **3× RRDB (Residual in Residual Dense Block)** — each RRDB contains 3 stacked Residual Dense Blocks (RDB)
- **Output conv layers** (no activation on the final layer)

Each **RDB** has 5 convolutional layers with densely connected feature maps and a residual scaling factor of 0.2. Activations alternate between LeakyReLU and PReLU.

### Discriminator
A standard convolutional discriminator with 8 conv layers (progressively increasing channels: 64 → 64 → 128 → 128 → 256 → 256 → 512 → 512) followed by two fully connected layers. Activations alternate between LeakyReLU and PReLU. Outputs a raw logit (no sigmoid — used with `BCEWithLogitsLoss`).

---

## Loss Functions

| Loss | Description |
|---|---|
| **Content Loss** | L1 loss between ResNet-18 feature embeddings of real and generated HR images |
| **Adversarial Loss** | Binary cross-entropy with logits (`BCEWithLogitsLoss`), weighted at 0.01 |
| **Total Generator Loss** | `content_loss + 0.01 × adversarial_loss` |
| **Discriminator Loss** | Average of real and fake BCE losses |

A pretrained ResNet-18 (with the classification head removed) is used as a fixed feature extractor for computing perceptual/content loss.

---

## Dataset

The dataset (`furniture/`) is expected to have the following structure:

```
furniture/
├── train/
│   ├── high_res/   # High-resolution training images
│   └── low_res/    # Corresponding low-resolution images
└── test/
    ├── high_res/   # High-resolution test images
    └── low_res/    # Corresponding low-resolution images
```

**Preprocessing:**
- Both high-res and low-res images are resized to **128×128**
- Random Gaussian blur (5×5 kernel) is applied to low-res training images with a 50% probability for augmentation
- An 80/20 train/validation split is applied automatically from the training directory

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning Rate | 0.0002 |
| Batch Size | 16 |
| Max Epochs | 500 |
| Early Stopping Patience | 20 epochs |
| LR Scheduler | `ReduceLROnPlateau` (factor=0.5, patience=20) |

Training stops early if validation loss does not improve for 20 consecutive epochs, and the best generator checkpoint is restored automatically.

---

## Evaluation

Model quality is evaluated using:

- **PSNR** (Peak Signal-to-Noise Ratio) — higher is better
- **SSIM** (Structural Similarity Index) — higher is better, computed with `win_size=3` and `multichannel=True`

Per-image PSNR and SSIM values are computed over the validation set, along with averages. The notebook also identifies the image with the highest PSNR and lowest SSIM.

---

## Output

- Training, discriminator, and validation loss curves are plotted across epochs
- Side-by-side comparisons of low-resolution input and super-resolved output are displayed
- Up to 400 super-resolved images can be saved to `msrgan_super_res_images/` as PNG files

---

## Requirements

```
torch
torchvision
opencv-python (cv2)
numpy
matplotlib
scikit-image
tqdm
```

Install with:

```bash
pip install torch torchvision opencv-python numpy matplotlib scikit-image tqdm
```

GPU acceleration is used automatically if CUDA is available (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## Usage

1. Prepare your dataset in the folder structure shown above.
2. Open `Modified_SRGAN_16th.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells in order:
   - Cells 1–2: Model and dataset definitions
   - Cell 3: Training
   - Cell 4: Loss visualization
   - Cell 5–6: PSNR/SSIM evaluation
   - Cell 7–8: Visual output and saving super-resolved images

---

## Notes

- The model name `MSRGAN` used in the code reflects the modifications made to the original SRGAN, primarily the use of RRDB blocks (borrowed from ESRGAN) and the mixed PReLU/LeakyReLU activation scheme.
- Both high-res and low-res images are currently resized to the same resolution (128×128), meaning the model learns quality enhancement rather than spatial upscaling. To enable upscaling, modify the `low_res_size` and add upsampling layers (e.g., pixel shuffle) to the generator.
