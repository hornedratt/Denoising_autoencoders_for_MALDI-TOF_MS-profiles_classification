from torch.utils.data import Dataset
import torch

class Customdataset (Dataset):
    def __init__ (self, profile, group, name):
        self.profile = torch.FloatTensor(profile)
        self.group = group
        self.name = name
        
    def __len__ (self):
        return int(self.profile.size(dim = 0))
    
    def __getitem__ (self, index):
        profile = self.profile[index, :]
        name = self.name[index]
        group = self.group[index]
        return{'profile': profile,
            'group': group,
            'ID': name}
    def cat (self, profile, group, name):
        self.profile = torch.cat((self.profile, profile), 0)
        self.group[len(self.group):] = group
        self.name[len(self.name):] = name