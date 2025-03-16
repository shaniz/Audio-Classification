import pickle

import torch
import torchvision
from torch.utils.data import *


class AudioDataset(Dataset):
    def __init__(self, pkl_dir, dataset_name, to_resize, transforms=None):
        self.data = []
        self.length = 1500 if dataset_name == "GTZAN" else 250
        self.transforms = transforms
        with open(pkl_dir, "rb") as f:
            self.data = pickle.load(f)
        self.resize = torchvision.transforms.Compose([torchvision.transforms.Resize(299)])
        self.to_resize = to_resize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # output_data = {}
        values = entry["values"].reshape(-1, 128, self.length)
        values = torch.Tensor(values)
        if self.transforms:
            values = self.transforms(values)
        if self.to_resize:
            values = self.resize(values)
        target = torch.LongTensor([entry["target"]])
        return values, target


def fetch_dataloader(pkl_dir, dataset_name, batch_size, num_workers, model):
    to_resize = True if model == 'inception' else False
    dataset = AudioDataset(pkl_dir, dataset_name, to_resize)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size,
                            num_workers=num_workers)  # , persistent_workers=True)  # Added persistent_workers
    return dataloader
