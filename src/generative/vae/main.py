import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from src.generative.vae import VAE, get_dataloader
from src.utils.metrics import AverageMeter
from src.utils.paths import checkpoint_path, data_dir, figure_dir, run_dir
from src.utils.visualization import plot_metric_curve, save_image_grid, save_reconstruction_comparison

DATA_DIR = data_dir("celeba")
CKPT_PATH = checkpoint_path("vae", "vae.pth")
WORK_DIR = run_dir("vae")
FIGURE_DIR = figure_dir("generative/vae")

epochs = 20
batch_size = 16 
lr = 1e-3
beta = 0.00025


def loss_fn(x, recon, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    return recon_loss + beta * kl_loss


def save_image(tensor, path):
    tensor = (tensor + 1) / 2  # Map [-1, 1] to [0, 1].
    img = ToPILImage()(tensor.clamp(0, 1))
    img.save(path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # DataLoader
    dataloader = get_dataloader(
        root=DATA_DIR,
        batch_size=batch_size,
        num_workers=0
    )

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print('Loaded checkpoint')

    # Train
    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter()

        for x in dataloader:
            x = x.to(device)

            recon, mu, logvar = model(x)
            loss = loss_fn(x, recon, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), x.size(0))

        avg_loss = loss_meter.avg
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}')

        torch.save(model.state_dict(), CKPT_PATH)

    plot_metric_curve(loss_history, FIGURE_DIR / "loss_curve.png", "VAE Training Loss", "Loss")

    # Reconstruct 
    model.eval()
    x = next(iter(dataloader)).to(device)
    recon, _, _ = model(x[:8])
    save_image(x[0].cpu(), WORK_DIR / "input.jpg")
    save_image(recon[0].cpu(), WORK_DIR / "recon.jpg")
    save_reconstruction_comparison(x.cpu(), recon.cpu(), FIGURE_DIR / "reconstruction_grid.png", n_samples=8)

    # Generate 
    samples = model.sample(device, n=8)
    save_image(samples[0].cpu(), WORK_DIR / "sample.jpg")
    save_image_grid(samples.cpu(), FIGURE_DIR / "sample_grid.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    main()
