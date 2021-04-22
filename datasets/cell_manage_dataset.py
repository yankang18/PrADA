import torch
from torch.utils.data import Dataset


class MultiPartitionDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data_list, label):
        self.data_list = data_list
        self.label = label

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, item_idx):
        tensor_list_i = []
        for data in self.data_list:
            tensor_i = torch.tensor(data[item_idx]).float()
            tensor_list_i.append(tensor_i)
        target_i = self.label[item_idx]
        return tensor_list_i, torch.tensor(target_i, dtype=torch.long)

