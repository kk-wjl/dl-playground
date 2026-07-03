from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image


def plot_metric_curve(values: list[float], save_path: Path, title: str, ylabel: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(values) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, values, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_image_grid(
    images: torch.Tensor,
    save_path: Path,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float, float] | None = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=value_range)
    save_image(grid, save_path)


def save_reconstruction_comparison(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    save_path: Path,
    n_samples: int = 8,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = min(n_samples, inputs.size(0), reconstructions.size(0))
    comparison = torch.cat([inputs[:n_samples], reconstructions[:n_samples]], dim=0)
    save_image_grid(
        comparison,
        save_path,
        nrow=n_samples,
        normalize=True,
        value_range=(-1, 1),
    )
