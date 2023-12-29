from abc import ABC

from torch.utils.data import Dataset


"""
    原生的torch Dataset
"""
class CustomDonutPretrainDataset(Dataset):
    def __init__(self, file_path):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass