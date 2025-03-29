import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse

import dataloaders.datasetaug
import dataloaders.datasetnormal
import models.densenet
import models.inception
import models.resnet
import train
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)

model_classes = {
    "densenet": models.densenet.DenseNet,
    "resnet": models.resnet.ResNet,
    "inception": models.inception.Inception,
}
columns = ["Model", "Dataset", "Pretrained", "Fold", "Acc", "Best Acc", "Best Acc - Epoch"]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    args = parser.parse_args()
    config_files = utils.list_files(args.config_path)

    for config_file in config_files:
        params = utils.Params(config_file)
        os.makedirs(os.path.dirname(params.results_path), exist_ok=True)
        utils.wrtie_to_csv(columns=columns, path=params.results_path)

        # Train and evaluate each fold
        for fold_num in range(1, params.num_folds + 1):
            print(f"\nWorking on {config_file} - fold {fold_num}:")
            pretrained_subdir = "pretrained" if params.pretrained else "random"
            log_subdir = f"{params.model}/{pretrained_subdir}/{params.dataset_name}/fold{fold_num}"
            writer = SummaryWriter(log_dir=f"{params.log_dir}/{log_subdir}", comment=f"{log_subdir}")

            train_loader = dataloaders.datasetaug.train_dataloader(params=params, fold_num=fold_num)
            val_loader = dataloaders.datasetaug.val_dataloader(params=params, fold_num=fold_num)
            model = model_classes[params.model](params.dataset_name, params.pretrained).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

            if params.pretrained:  # Different scheduler for pretrained/random initialization
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            else:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, train.lr_lambda)

            acc, best_acc, best_acc_epoch = train.train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn,
                                                     writer,
                                                     params, fold_num, scheduler)

            utils.wrtie_to_csv(data=[params.model, params.dataset_name, params.pretrained, fold_num, acc, best_acc, best_acc_epoch],
                               columns=columns, path=params.results_path)
            print(
                f"Saved results for {params.model} | {params.dataset_name} | pretrained- {params.pretrained} | fold {fold_num}")
