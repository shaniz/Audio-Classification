### Advanced Topics in Audio Processing using Deep Learning - Final Project Report:
## Pretrained CNN models for Audio Classification
##### Based on the paper “Rethinking CNN Models for Audio Classification”
##### Submitted by: Daya Matok Gawi (ID 311143051) and Shani Zola (ID 206361909)

This project contains the PyTorch code and the report for reproducing and extending the
paper [Rethinking CNN Models for Audio Classification](https://arxiv.org/abs/2007.11154). 

### Original Repo from Paper
[original-repo-from-paper](https://github.com/kamalesh0406/Audio-Classification.git)

### Our Repo
[our-repo](https://github.com/shaniz/Audio-Classification.git)

### Python Version
Python 3.10.

### Dataset:
The experiments are conducted on the dataset ESC-50 which can be downloaded from the link:
[ESC-50](https://github.com/karolpiczak/ESC-50) or by running the script get_esc_50.py 

### Preprocessing
The preprocessing is done separately to save time during the training of the models.
It creates 5 folds of spectrogram inputs for training and evaluation.
For ESC-50:
```console
python preprocessing/preprocessingESC.py --csv_file path/to/file.csv --data_dir path/to/audio_data/ --store_dir spectrograms/esc/ --sampling_rate 44100
```
After running the script mentioned in the Dataset section (the final directory is attached):
```console
--data_dir -> ESC-50/audio
--csv_file -> ESC-50/meta/esc50.csv
```
So the preprocessing command should be:
```console
python preprocessing/preprocessingESC.py --csv_file ESC-50/meta/esc50.csv --data_dir ESC-50/audio --store_dir spectrograms/esc/ --sampling_rate 44100
```

### Experiment parts
We organized the experiments according to the parts of the paper, the corresponding file names are:

B1. Single model Comparison - B1.py

B3. Analysis:
1. Weights Change (SVCCA) - B3-analysis_svcca.py
2. Weight Fusion - B3-analysis_pretrained.py
3. Weights Freeze - B3-analysis_pretrained.py
4. Model Cutoff - B3-analysis_pretrained.py

C. Deep ensemble - C-deep_ensemble.py

### Training the Models
The configurations for training the models are provided in the config folder. 
The sample_config.json explains the details of all the variables in the configurations. 
The command for training each part of the paper (B1, B3-analysis_pretrained, C) is:
```console
python <part-file-name> --config_path config/your_config.json
```

You can also give a folder for working on all the config files under it, for example:
```console
python B3-analysis_pretrained.py --config_path config/B3/densenet/weight_fusion
```

The training process (loss+accuracy) is saved in the 'runs' directory.
You can see the training process by running:
```console
tensorboard --logdir=<log-directory>
```

The command for getting B3-analysis_svcca results is:
```console
python B3-analysis_svcca.py
  --config_pre <Path-to-the-pretrained-model-config-file>
  --config_rnd <Path-to-the-pretrained-model-config-file>
  --ckpt_pre <Checkpoint-path-of-pretrained-fine-tuned-model>
  --ckpt_rnd <Checkpoint-path-of-random-weights-trained-model>
```

### Folders Structure
For folders: checkpoints, config, runs (tensorBoard):
```console
<folder-name>/<part-in-the-paper>/<model-name>/<weights>/<dataset>
```

### Results
Appear in the 'results' folder:
```console
results/<part-in-the-paper>/<part-in-the-paper+model>
```
