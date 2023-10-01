import pandas as pd
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

    # def cat (self, profile, group, name):
    #     self.profile = torch.cat((self.profile, profile), 0)
    #     self.group[len(self.group):] = group
    #     self.name[len(self.name):] = name
