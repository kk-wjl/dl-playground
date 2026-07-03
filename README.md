# DL Playground

This repository is a small PyTorch playground for studying classical models, transformers, and generative models with a clean package layout and shared project interfaces.

## Project Structure

```text
dl-playground/
├── artifacts/
│   ├── checkpoints/          # Saved model weights
│   ├── figures/              # Diagrams, demos, and result images
│   └── runs/                 # Generated outputs and reconstruction samples
├── data/
│   ├── celeba/               # CelebA images
│   ├── cifar10/              # CIFAR-10 
│   └── mnist/                # MNIST data
├── src/
│   ├── classical/
│   │   ├── cnn/
│   │   ├── rnn/
│   │   └── softmax/
│   ├── generative/
│   │   ├── autoregressive/
│   │   ├── diffusion/
│   │   ├── gan/
│   │   └── vae/
│   ├── transformer/
│   └── utils/
└── .gitignore
```

## Package Conventions

- All source code lives under `src/`.
- Each model directory is a Python package and includes `__init__.py`.
- Entry scripts should be run as modules from the project root.
- Internal imports should use package imports instead of same-folder script imports.

## Shared Interfaces

Path handling is centralized in `src/utils/paths.py`.

- `data_dir(name)`: returns a dataset directory under `data/`
- `checkpoint_path(experiment, filename)`: returns a checkpoint path under `artifacts/checkpoints/`
- `run_dir(experiment)`: returns a run output directory under `artifacts/runs/`
- `figure_dir(experiment)`: returns a figure directory under `artifacts/figures/`
- `src/utils/metrics.py`: shared metric helpers such as running averages and accuracy
- `src/utils/visualization.py`: shared plotting and image export helpers

This keeps dataset, checkpoint, and output paths out of individual experiment scripts.

## Running Entry Points

Run modules from the repository root with `python -m`:

```bash
uv run python -m src.classical.cnn.train
uv run python -m src.classical.rnn.train
uv run python -m src.generative.vae.main
uv run python -m src.generative.diffusion.train
uv run python -m src.transformer.train
```

## Project Rules

- Store datasets only in `data/`.
- Store checkpoints and generated outputs only in `artifacts/`.
- Do not keep cache files such as `__pycache__/`, `.DS_Store`, or `.ipynb_checkpoints/`.
- Avoid creating ad hoc local output folders inside source packages.

## References

- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- VAE: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- GAN: [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- Diffusion: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

## Notes

- If you add new experiments, follow the same `src/ + data/ + artifacts/` layout instead of introducing local data or output folders.
