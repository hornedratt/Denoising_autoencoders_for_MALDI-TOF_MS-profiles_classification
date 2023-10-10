import pandas as pd
import torch
from typing import List, Union

from torch import FloatTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataSet(Dataset):
    def __init__(self,
                 profile: pd.DataFrame,
                 group: pd.DataFrame,
                 name: pd.DataFrame):
        self.profile = FloatTensor(profile)
        self.group = group
        self.name = name

    def __len__(self):
        return int(self.profile.size(dim = 0))

    def __getitem__(self, idx: int) -> tuple[Union[FloatTensor, str], Union[FloatTensor, str]]:
        return self.profile[idx, :], self.group[idx], self.name[idx]

def collate_fn(batch_objs: List[Union[FloatTensor, str]]):
    profiles = []
    groups = []
    IDs = []
    for elem in batch_objs:
        profile, group, ID = elem

        profiles.append(profile)
        groups.append(group)
        IDs.append(ID)
    profiles = torch.stack(profiles)
    groups = torch.stack(groups)
    IDs = torch.stack(IDs)
    return profiles, groups, IDs