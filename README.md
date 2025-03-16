# Rethinking CNN Models for Audio Classification

This repository contains the PyTorch code for our
paper [Rethinking CNN Models for Audio Classification](https://arxiv.org/abs/2007.11154). The experiments are conducted
on the following three datasets which can be downloaded from the links provided:

1. [ESC-50](https://github.com/karolpiczak/ESC-50)
2. [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
3. [GTZAN](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

### Preprocessing

The preprocessing is done separately to save time during the training of the models.

For ESC-50:

```console
python preprocessing/preprocessingESC.py --csv_file /path/to/file.csv --data_dir /path/to/audio_data/ --store_dir spectrograms/esc/ --sampling_rate 44100
```

For UrbanSound8K:

```console
python preprocessing/preprocessingUSC.py --csv_file /path/to/file.csv --data_dir /path/to/audio_data/ --store_dir spectrograms/usc/ --sampling_rate 22050
```

For GTZAN:

```console
python preprocessing/preprocessingGTZAN.py --data_dir /path/to/audio_data/ --store_dir spectrograms/gtzan/ --sampling_rate 22050
```

### Training the Models

The configurations for training the models are provided in the config folder. The sample_config.json explains the
details of all the variables in the configurations. The command for training is:

```console
python train.py --config_path /config/your_config.json
```

### Folders Structure
For folders: checkpoints, config, runs (tensorBoard):

```console
<part-in-the-paper>/<model-name>/<weights>/<dataset>
```

### Number of Parameters
Densnet201 - 20.2M
Inceptionv3 - 24M
Resnet50 - 25.6M