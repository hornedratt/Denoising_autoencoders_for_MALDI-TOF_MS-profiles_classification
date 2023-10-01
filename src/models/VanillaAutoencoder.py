import torch.nn as nn
from torch import FloatTensor


class VanillaAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12001, 6000),
            nn.ReLU(),
            nn.Linear(6000, 750),
            nn.ReLU(),
            nn.Linear(750, 50)
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(50, 750),
            nn.ReLU(),
            nn.Linear(750, 6000),
            # nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(6000, 12001)
        )

    def forward(self, x: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        x = self.encoder(x)
        embadding = x.copy()
        reconstruction = self.decoder(x)
        return reconstruction, embadding
