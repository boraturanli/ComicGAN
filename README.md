# ComicGAN

A modified Pix2Pix architecture for translating facial photographs into comic-style images. Built as part of the Machine Learning course at Vrije Universiteit Amsterdam.

**Full paper:** [`comicgan.pdf`](./Optimizing-Pix2Pix-for-Comics.pdf)  
**Kaggle notebook:** [kaggle.com/code/rohanshreyasmenon/comicgan](https://www.kaggle.com/code/rohanshreyasmenon/comicgan)

---

## What it does

Given a portrait photo as input, the model generates a comic-book stylised version of it. The core architecture follows [Pix2Pix (Isola et al., 2016)](https://arxiv.org/abs/1611.07004), with an extended U-Net generator (8 downsampling layers + residual blocks at the bottleneck) and a PatchGAN discriminator.

The main research question was: **how does the weighting between adversarial loss and pixel-wise reconstruction loss (L1) affect output quality and training stability?**

![Example outputs](./comicGAN_figures_fixed.pdf)

---

## Architecture

**Generator:** U-Net with residual blocks  
- 8 encoder (downsampling) layers, compressing input to a 512×2×2 bottleneck  
- 4 residual blocks at the bottleneck  
- 8 decoder (upsampling) layers with skip connections from encoder  
- Tanh activation on output; dropout in deeper decoder layers  

**Discriminator:** PatchGAN  
- Takes the concatenated input + generated image (6 channels) as input  
- 4 convolutional layers with LeakyReLU; sigmoid output  
- Evaluates realism in local patches rather than the full image  

**Loss function**

$$\mathcal{L}_G = \mathcal{L}_{GAN} + \lambda \cdot \mathcal{L}_{L1}$$

The paper investigates different strategies for setting λ.

---

## Key findings

**Static λ outperforms dynamic λ.** Fixing λ creates an implicit annealing effect: early in training the L1 term dominates and guides the generator toward structurally accurate outputs; as training progresses and L1 loss decreases, the adversarial term naturally takes over and refines fine details.

**Dynamic λ** (which recalculates λ at each step to maintain a fixed ratio between the two losses) disrupts this natural dynamic. Because adversarial loss stays roughly constant while L1 decreases, dynamic λ grows proportionally to 1/L1, progressively over-weighting the adversarial term and causing artifacts similar to GAN-only training.

**Best static value: λ = 6.75** (≈50% L1 contribution). Produces the sharpest, most comic-like results with well-preserved proportions.

---

## Setup

```bash
pip install torch torchvision numpy matplotlib tqdm pandas pillow scikit-learn
```

Dataset: [Comic Faces (Paired, Synthetic) v2](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic-v2) — 10,000 paired portrait/comic images at 1024×1024, resized to 512×512 for training.

Training was run on Kaggle with 2x NVIDIA T4 GPUs. Batch size 48, Adam optimiser (lr=0.0002, β1=0.5, β2=0.999), 50 epochs.

---

## Authors

Rohan Menon, Bora Turanli Lledó, Terrence Semeleer, Philip Versluis, Vencel Kuba  
Vrije Universiteit Amsterdam, Group 129
