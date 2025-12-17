"""
Image Preprocessing Techniques in Python (Sample Dataset)

This script demonstrates common image preprocessing techniques on a sample dataset:
- Load & visualize images (sklearn digits dataset)
- Resize images (8x8 -> 32x32)
- Normalize / standardize pixel values
- Denoise (median / Gaussian filters)
- Enhance contrast (histogram equalization / CLAHE)
- Threshold (Otsu)
- Data augmentation (rotate/shift/flip/noise)
- Prepare data for deep learning (PyTorch Dataset/DataLoader)

Run:
    python image_preprocessing_sample_digits.py

If running in a fresh environment, install dependencies:
    pip install numpy matplotlib scikit-image scikit-learn scipy torch torchvision
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage import exposure, util

from scipy.ndimage import median_filter, gaussian_filter, shift, rotate

import torch
from torch.utils.data import Dataset, DataLoader


def show_grid(images, labels=None, n=12, cols=6, title=None):
    """Show a grid of images using matplotlib."""
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
        if labels is not None:
            plt.title(str(labels[i]))
    if title:
        plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()


def augment(img: np.ndarray) -> np.ndarray:
    """Simple classical augmentation pipeline for grayscale images in [0,1]."""
    rot = rotate(img, angle=np.random.uniform(-20, 20), reshape=False, mode="nearest")
    sh = shift(rot, shift=(np.random.uniform(-3, 3), np.random.uniform(-3, 3)), mode="nearest")
    if np.random.rand() < 0.5:
        sh = np.fliplr(sh)
    sh = sh + np.random.normal(0, 0.03, size=sh.shape)
    sh = np.clip(sh, 0, 1)
    return sh.astype(np.float32)


class DigitsDataset(Dataset):
    """PyTorch Dataset with optional augmentation."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, do_augment: bool = False):
        self.images = images
        self.labels = labels
        self.do_augment = do_augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.do_augment:
            img = augment(img)
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        y_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_t, y_t


def main():
    np.random.seed(42)

    # 1) Load sample dataset
    digits = load_digits()
    X = digits.images  # (n, 8, 8), values 0..16
    y = digits.target

    print("Raw shapes:", X.shape, y.shape)
    show_grid(X, y, n=12, cols=6, title="Raw digits (8×8 grayscale)")

    # 2) Resize to 32x32
    target_size = (32, 32)
    X_resized = np.array([resize(img, target_size, anti_aliasing=True) for img in X])
    print("Resized shape:", X_resized.shape)
    show_grid(X_resized, y, n=12, cols=6, title="Resized digits (32×32)")

    # 3) Normalize to [0,1] and standardize (dataset-level)
    X_norm = X_resized.astype(np.float32)
    X_norm = (X_norm - X_norm.min()) / (X_norm.max() - X_norm.min() + 1e-8)
    mean = float(X_norm.mean())
    std = float(X_norm.std() + 1e-8)
    X_std = (X_norm - mean) / std

    print("Normalized range:", float(X_norm.min()), float(X_norm.max()))
    print("Standardized mean/std:", mean, std)
    show_grid(X_norm, y, n=12, cols=6, title="Normalized [0,1]")

    # 4) Denoising demo (add synthetic salt & pepper noise)
    idx = 0
    img = X_norm[idx]
    noisy = util.random_noise(img, mode="s&p", amount=0.15)
    den_med = median_filter(noisy, size=3)
    den_gauss = gaussian_filter(noisy, sigma=1.0)

    plt.figure(figsize=(10, 3))
    for i, (im, t) in enumerate(
        [(img, "Original"), (noisy, "Noisy (s&p)"), (den_med, "Median filtered"), (den_gauss, "Gaussian filtered")]
    ):
        plt.subplot(1, 4, i + 1)
        plt.imshow(im, cmap="gray")
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 5) Contrast enhancement
    img = X_norm[1]
    eq = exposure.equalize_hist(img)
    clahe = exposure.equalize_adapthist(img, clip_limit=0.03)

    plt.figure(figsize=(9, 3))
    for i, (im, t) in enumerate([(img, "Original"), (eq, "Hist. Equalization"), (clahe, "CLAHE")]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(im, cmap="gray")
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 6) Thresholding (Otsu)
    img = X_norm[2]
    t = threshold_otsu(img)
    binary = (img > t).astype(np.float32)

    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binary, cmap="gray")
    plt.title(f"Otsu threshold (t={t:.3f})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 7) Augmentation preview
    img = X_norm[3]
    augs = [augment(img) for _ in range(5)]

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 6, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for i in range(5):
        plt.subplot(1, 6, i + 2)
        plt.imshow(augs[i], cmap="gray")
        plt.title(f"Aug {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 8) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)
    print("Train/Test:", X_train.shape, X_test.shape)

    # 9) PyTorch dataset/loader
    train_ds = DigitsDataset(X_train, y_train, do_augment=True)
    test_ds = DigitsDataset(X_test, y_test, do_augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    images, labels = next(iter(train_loader))
    print("One batch:", images.shape, labels.shape)

    # Visualize a batch
    batch = images[:12].squeeze(1).numpy()
    show_grid(batch, labels[:12].numpy(), n=12, cols=6, title="Augmented training batch (32×32)")

    print("\nDone. (Tip: You can adapt this pipeline to your prescription/radiology datasets.)")


if __name__ == "__main__":
    main()
