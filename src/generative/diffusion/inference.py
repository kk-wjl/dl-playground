import torch

from src.generative.diffusion.ddpm import DDPM
from src.generative.diffusion.model import UNet
from src.utils.paths import checkpoint_path

def main():
    data_dim = 11
    cond_dim = 5
    batch = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model = UNet(data_dim=data_dim, cond_dim=cond_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path("diffusion", "weights.pth"), map_location=device))
    model.eval()
    ddpm = DDPM(timesteps=1000, device=device)

    cond = torch.randn(batch, cond_dim, device=device) # replace with actual condition

    # Reverse diffusion process
    x = torch.randn(batch, data_dim, device=device) # start from pure noise
    for t in reversed(range(ddpm.timesteps)): # reverse diffusion process
        t_batch = torch.full((batch,), t, device=device, dtype=torch.long) # current timestep
        x = ddpm.p_sample(model, x, t_batch, cond) # sample x_t-1 from x_t and noise prediction
    print('Sampled x:', x) # final denoised sample

if __name__ == '__main__':
    main()
