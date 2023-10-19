import pandas as pd
import torch
import numpy as np
from typing import List, Union

from torch import FloatTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataSet(Dataset):
    def __init__(self,
                 profile: np.array,
                 group: pd.DataFrame,
                 name: pd.DataFrame):
        self.profile = FloatTensor(profile)
        self.group = group
        self.name = name

    def __len__(self):
        return int(self.profile.size(dim = 0))

    def __getitem__(self, idx: int) -> tuple[Union[FloatTensor, str], Union[FloatTensor, str]]:
        return self.profile[idx, :], self.group[idx], self.name[idx]

    def subset(self, indices: list) -> tuple[np.array, pd.Series, pd.Series]:
        profiles = self.profile[indices, :]
        groups = self.group.iloc[indices].reset_index(drop=True)
        names = self.name.iloc[indices].reset_index(drop=True)
        return profiles, groups, names

def collate_fn(batch_objs: List[Union[FloatTensor, str]], noise_factor: float):
    """collate_fn для denoising автоенкодера, создает зашусленный батч и возвращает его с чистым
    :param batch_objs: сам батч
    :param noise_factor: уровень генирируемого шума
    :return: чистый батч, зашумленный батч, метки группы, метки штамма
    """
    profiles_noise = []
    profiles = []
    groups = []
    IDs = []
    for elem in batch_objs:
        profile, group, ID = elem

        noise = profile + \
                torch.FloatTensor(np.random.normal(loc=0.0,
                                                   scale=noise_factor * profile,
                                                   size=list(profile.size())))
        noise = torch.abs(noise)
        y = torch.ones(list(profile.size()))
        noise = torch.where(noise < 1, noise, y)

        profiles_noise.append(noise)
        profiles.append(profile)
        groups.append(group)
        IDs.append(ID)
    profiles_noise = torch.stack(profiles_noise)
    profiles = torch.stack(profiles)
    # groups = torch.stack(groups)
    # IDs = torch.stack(IDs)
    return profiles, profiles_noise, groups, IDs