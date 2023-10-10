import torch
import pandas as pd
import click
import progressbar as pb
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.VanillaAutoencoder import VanillaAutoencoder
from src.data.CustomDataSet import CustomDataSet

def train_one_set(n_epochs: int = 50,
                  lr: float = 0.001,
                  noise_factor: float = 40,
                  L=F.mse_loss) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    noise_factor = noise_factor / 100
    autoencoder = VanillaAutoencoder().to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    train_set = pd.read_csv(os.path.join('..', '..', 'data\\processed\\original_MS_profiles.csv'), sep=';')
    train_set = CustomDataSet(train_set.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                              train_set['group'],
                              train_set['ID'])
    train_loader = DataLoader(train_set, batch1)
    embaddings = torch.Tensor()
    truth = torch.Tensor()
    pred = torch.Tensor()

    for epoch in pb.progressbar(range(n_epochs)):
        autoencoder.train()
        train_losses_per_epoch = []

        for X_batch in train_data:
            noise = X_batch['profile'] + \
                    torch.FloatTensor(np.random.normal(loc=0.0, \
                                                       scale=noise_factor * X_batch['profile'],
                                                       size=list(X_batch['profile'].size())))  # шумим
            noise = torch.abs(noise)
            y = torch.ones(list(X_batch['profile'].size()))
            noise = torch.where(noise < 1, noise, y)

            X_batch['profile'] = X_batch['profile'].to(device)  # чистые векторы
            noise = noise.to(device)

            optimizer.zero_grad()
            reconstructed, embadding = self.autoencoder.forward(noise)  # скармливаем шум
            loss = L(reconstructed, X_batch['profile'])  # сравниваем с читыми
            loss.backward()
            optimizer.step()
            train_losses_per_epoch.append(loss.item())

        train_losses.append(np.mean(train_losses_per_epoch))
        self.train_losses = train_losses

        self.autoencoder.eval()
        val_losses_per_epoch = []
        with torch.no_grad():
            for X_batch in self.train_data:
                noise = X_batch['profile'] + \
                        torch.FloatTensor(np.random.normal(loc=0.0, \
                                                           scale=noise_factor * X_batch['profile'],
                                                           size=list(X_batch['profile'].size())))  # шумим
                noise = torch.abs(noise)
                y = torch.ones(list(X_batch['profile'].size()))
                noise = torch.where(noise < 1, noise, y)

                noise = noise.to(device)
                X_batch['profile'] = X_batch['profile'].to(device)

                reconstructed, embadding = self.autoencoder(noise)
                loss = L(reconstructed, X_batch['profile'])
                val_losses_per_epoch.append(loss.item())

                embadding = embadding.to('cpu')
                reconstructed = reconstructed.to('cpu')
                X_batch['profile'] = X_batch['profile'].to('cpu')

                self.embaddings = customdataset(embadding, X_batch['group'].copy(), X_batch['ID'].copy())
                self.truth = customdataset(X_batch['profile'], X_batch['group'].copy(), X_batch['ID'].copy())
                self.pred = customdataset(reconstructed, X_batch['group'].copy(), X_batch['ID'].copy())

        val_losses.append(np.mean(val_losses_per_epoch))
        self.val_losses = val_losses
        return None