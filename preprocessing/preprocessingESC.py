import argparse
import pickle as pkl
import os

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--store_dir", type=str)
parser.add_argument("--sampling_rate", default=44100, type=int)


def extract_spectrogram(values, clip, entries):
    for data in entries:

        num_channels = 3
        window_sizes = [25, 50, 100]
        hop_sizes = [10, 25, 50]
        centre_sec = 2.5

        specs = []
        for i in range(num_channels):
            window_length = int(round(window_sizes[i] * args.sampling_rate / 1000))
            hop_length = int(round(hop_sizes[i] * args.sampling_rate / 1000))

            clip = torch.Tensor(clip)
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=4410,
                                                        win_length=window_length, hop_length=hop_length, n_mels=128)(
                clip)
            eps = 1e-6
            spec = spec.numpy()
            spec = np.log(spec + eps)
            spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
            specs.append(spec)
        new_entry = dict()
        new_entry["audio"] = clip.numpy()
        new_entry["values"] = np.array(specs)
        new_entry["target"] = data["target"]
        values.append(new_entry)


def extract_features(audios):
    audio_names = list(audios.filename.unique())
    values = []
    for audio in audio_names:
        clip, sr = librosa.load("{}/{}".format(args.data_dir, audio), sr=args.sampling_rate)
        entries = audios.loc[audios["filename"] == audio].to_dict(orient="records")
        extract_spectrogram(values, clip, entries)
        print("Finished audio {}".format(audio))
    return values


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.store_dir, exist_ok=True)

    audios = pd.read_csv(args.csv_file, skipinitialspace=True)
    num_folds = 5

    for i in range(1, num_folds + 1):
        training_audios = audios.loc[audios["fold"] != i]
        validation_audios = audios.loc[audios["fold"] == i]

        training_values = extract_features(training_audios)
        with open("{}training128mel{}.pkl".format(args.store_dir, i), "wb") as handler:
            pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

        validation_values = extract_features(validation_audios)
        with open("{}validation128mel{}.pkl".format(args.store_dir, i), "wb") as handler:
            pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)
