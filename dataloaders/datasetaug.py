import pickle

import librosa
import numpy as np
import torch
import torchaudio
import torchvision
from PIL import Image
from torch.utils.data import *


class MelSpectrogram(object):
    def __init__(self, bins, mode, dataset):
        self.window_length = [25, 50, 100]
        self.hop_length = [10, 25, 50]
        self.fft = 4410 if dataset == "ESC" else 2205
        self.melbins = bins
        self.mode = mode
        self.sr = 44100 if dataset == "ESC" else 22050
        self.length = 1500 if dataset == "GTZAN" else 250

    def __call__(self, value):
        sample = value
        limits = ((-2, 2), (0.9, 1.2))

        if self.mode == "train":
            pitch_shift = np.random.randint(limits[0][0], limits[0][1] + 1)
            time_stretch = np.random.random() * (limits[1][1] - limits[1][0]) + limits[1][0]
            # new_audio = librosa.effects.time_stretch(librosa.effects.pitch_shift(sample, self.sr, pitch_shift), time_stretch)
            new_audio = librosa.effects.time_stretch(
                librosa.effects.pitch_shift(sample, sr=self.sr, n_steps=pitch_shift), rate=time_stretch)
        else:
            # pitch_shift = 0
            # time_stretch = 1
            new_audio = sample
        specs = []
        for i in range(len(self.window_length)):
            clip = torch.Tensor(new_audio)

            window_length = int(round(self.window_length[i] * self.sr / 1000))
            hop_length = int(round(self.hop_length[i] * self.sr / 1000))
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=self.fft, win_length=window_length,
                                                        hop_length=hop_length, n_mels=self.melbins)(clip)
            eps = 1e-6
            spec = spec.numpy()
            spec = np.log(spec + eps)
            spec = np.asarray(torchvision.transforms.Resize((128, self.length))(Image.fromarray(spec)))
            specs.append(spec)
        specs = np.array(specs).reshape(-1, 128, self.length)
        specs = torch.Tensor(specs)
        return specs


class AudioDataset(Dataset):
    def __init__(self, pkl_dir, dataset_name, to_resize, transforms=None):
        self.transforms = transforms
        self.data = []
        self.length = 1500 if dataset_name == "GTZAN" else 250
        with open(pkl_dir, "rb") as f:
            self.data = pickle.load(f)
        self.resize = torchvision.transforms.Compose([torchvision.transforms.Resize(299)])
        self.to_resize = to_resize

    def __len__(self):
        if self.transforms.mode == "train":
            return 2 * len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            new_idx = idx - len(self.data)
            entry = self.data[new_idx]
            if self.transforms:
                values = self.transforms(entry["audio"])
        else:
            entry = self.data[idx]
            values = torch.Tensor(entry["values"].reshape(-1, 128, self.length))

        if self.to_resize:
            values = self.resize(values)

        target = torch.LongTensor([entry["target"]])
        return values, target


def fetch_dataloader(pkl_dir, dataset_name, batch_size, num_workers, mode, model):
    transforms = MelSpectrogram(128, mode, dataset_name)
    to_resize = True if model == 'inception' else False
    dataset = AudioDataset(pkl_dir, dataset_name, to_resize, transforms=transforms)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size,
                            num_workers=num_workers)  # , persistent_workers=True)
    return dataloader


def train_dataloader(params, fold_num):
    train_loader = fetch_dataloader(
        "{}training128mel{}.pkl".format(params.data_dir, fold_num), params.dataset_name, params.batch_size,
        params.num_workers, 'train', params.model)
    return train_loader


def val_dataloader(params, fold_num):
    val_loader = fetch_dataloader(
        "{}validation128mel{}.pkl".format(params.data_dir, fold_num), params.dataset_name, params.batch_size,
        params.num_workers, 'validation', params.model)
    return val_loader
