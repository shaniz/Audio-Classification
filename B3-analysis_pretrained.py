import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse

import dataloaders.datasetaug
import dataloaders.datasetnormal
import models.densenet
import train
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)

model_classes = {
    "densenet/weight_fusion": models.densenet.DenseNetWeightFusion,
    "densenet/weight_freeze": models.densenet.DenseNetWeightFreeze,
    "densenet/cutoff": models.densenet.DenseNetModelCutoff
}
columns = ["Model", "Dataset", "Fold", "Layer", "Acc", "Best Acc", "Best Acc - Epoch"]
layers_by_model = {
    "densenet/weight_fusion": ["conv0", "denseblock1", "denseblock2", "denseblock3", "denseblock4"],
    "densenet/weight_freeze": ["conv0", "denseblock1", "denseblock2", "denseblock3", "denseblock4"],
    "densenet/cutoff": ["denseblock3", "denseblock4"]
}


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
            train_loader = dataloaders.datasetaug.train_dataloader(params=params, fold_num=fold_num)
            val_loader = dataloaders.datasetaug.val_dataloader(params=params, fold_num=fold_num)

            layers = layers_by_model[params.model]
            for layer in layers:
                print(f"\nWorking on {config_file} - layer {layer} - fold {fold_num}:")

                log_subdir = f"{params.model}/{params.dataset_name}/fold{fold_num}/{layer}"
                writer = SummaryWriter(log_dir=f"{params.log_dir}/{log_subdir}", comment=f"{log_subdir}")

                model = model_classes[params.model](params.dataset_name, layer, params.pretrained).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
                # Only pretrained here
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                acc, best_acc, best_acc_epoch = train.train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn,
                                                         writer,
                                                         params, fold_num, scheduler, layer=layer)

                utils.wrtie_to_csv(
                    data=[params.model, params.dataset_name, fold_num, layer, acc, best_acc, best_acc_epoch],
                    columns=columns, path=params.results_path)
                print(
                    f"Saved results for {params.model} | {params.dataset_name} | fold {fold_num} | layer {layer}")
