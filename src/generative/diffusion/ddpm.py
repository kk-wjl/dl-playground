import torch

class DDPM:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t):
        """
        Forward diffusion (q): sample x_t from x0 at timestep t
        Args:
            x0: [B, data_dim] - clean data
            t: [B] - timestep
        Returns:
            x_t: [B, data_dim] - noisy data
            noise: [B, data_dim] - the noise added
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).unsqueeze(1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).unsqueeze(1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    def p_sample(self, model, x, t, cond):
        """
        Reverse diffusion (p): sample x_{t-1} from x_t using model prediction
        Args:
            model: noise prediction network
            x: [B, data_dim] - current noisy data
            t: [B] - current timestep
            cond: [B, cond_dim] - condition
        Returns:
            x_{t-1}: [B, data_dim] - denoised sample for previous step
        """
        noise_pred = model(x, t, cond)
        sqrt_recip_alpha = 1.0 / torch.sqrt(self.alphas[t]).unsqueeze(1)
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas[t]).unsqueeze(1)
        x0_pred = sqrt_recip_alpha * (x - sqrt_one_minus_alpha * noise_pred)
        beta_t = self.betas[t].unsqueeze(1)
        mean = torch.sqrt(self.alphas[t]).unsqueeze(1) * x0_pred + (1 - torch.sqrt(self.alphas[t]).unsqueeze(1)) * x
        if t[0] > 0:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(beta_t) * noise
        else:
            return mean