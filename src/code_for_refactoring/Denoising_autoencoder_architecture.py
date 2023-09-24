import torch
import torch.nn as nn

class Vanilla_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12001, 6000),
            nn.ReLU(),
            #nn.Dropout(p=0.25),
            nn.Linear(6000, 750),
            nn.ReLU()
            )
        self.fc = nn.Linear(750, 50)
        
        self.unfc = nn.Linear(50, 750)
        
        self.decoder = nn.Sequential(
        nn.ReLU(), 
        nn.Linear(750, 6000),
        #nn.Dropout(p=0.25),
        nn.ReLU(),
        nn.Linear(6000, 12001)
        )
              
    def forward(self, x):
        x = self.encoder(x)
        embadding = self.fc(x)
        x = self.unfc(embadding)
        reconstruction = self.decoder(x)

        return reconstruction, embadding