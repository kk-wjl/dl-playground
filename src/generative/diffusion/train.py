import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.generative.diffusion.ddpm import DDPM
from src.generative.diffusion.model import UNet
from src.utils.paths import checkpoint_path

def main():
    # dataset setup
    data_dim = 11
    cond_dim = 5
    x = torch.randn(10000, data_dim)
    cond = torch.randn(10000, cond_dim)
    dataset = TensorDataset(x, cond)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # model, ddpm, optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(data_dim=data_dim, cond_dim=cond_dim).to(device)
    ddpm = DDPM(timesteps=1000, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, cond in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            # x: [B, data_dim], cond: [B, cond_dim], t: [B]
            x = x.to(device)
            cond = cond.to(device)
            t = torch.randint(0, ddpm.timesteps, (x.size(0),), device=device)

            x_t, noise = ddpm.q_sample(x, t) # forward diffusion: get x_t and the true noise
            pred_noise = model(x_t, t, cond) # predict noise with the model
            loss = nn.MSELoss()(pred_noise, noise) # MSE loss between predicted and true noise

            optimizer.zero_grad()
            loss.backward() # backpropagation
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), checkpoint_path("diffusion", "weights.pth"))

if __name__ == '__main__':
    main()
