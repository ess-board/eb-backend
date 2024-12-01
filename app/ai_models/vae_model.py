import torch
from torch import nn

# VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 인코더
        self.enc_fc = nn.Linear(64 * 64, 400)
        self.mu_fc = nn.Linear(400, 20)
        self.logvar_fc = nn.Linear(400, 20)

        # 디코더
        self.dec_fc1 = nn.Linear(20, 400)
        self.dec_fc2 = nn.Linear(400, 64 * 64)

    def encode(self, x):
        h = torch.relu(self.enc_fc(x))
        return self.mu_fc(h), self.logvar_fc(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_fc2(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 64 * 64))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x.view(-1, 64 * 64), reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss
