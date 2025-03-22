import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import dataloaders.datasetaug
import dataloaders.datasetnormal
import models.densenet
import models.inception
import models.resnet
import train
import utils
import validate


class DeepEnsemble(nn.Module):
    def __init__(self, models):
        super(DeepEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models],
                              dim=0)  # Shape: (num_models, batch_size, num_classes)
        ensemble_output = torch.mean(outputs, dim=0)  # Averaging predictions
        return ensemble_output


config_path = "config/C/resnet"
results_path = "results/C-resnet.csv"
os.makedirs(os.path.dirname(results_path), exist_ok=True)
model_classes = {
    "densenet": models.densenet.DenseNet,
    "resnet": models.resnet.ResNet,
    "inception": models.inception.Inception,
}
log_dir = "runs/C"
columns = ["Model", "Model Number", "Dataset", "Pretrained", "Fold", "Acc", "Best Acc", "Best Acc - Epoch"]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    utils.wrtie_to_csv(columns=columns, path=results_path)
    loss_fn = nn.CrossEntropyLoss()
    config_files = utils.list_files(config_path)

    for config_file in config_files:
        params = utils.Params(config_file)
        # Working on the same fold for all models
        train_loader = dataloaders.datasetaug.train_dataloader(params=params, fold_num=params.fold_num)
        val_loader = dataloaders.datasetaug.val_dataloader(params=params, fold_num=params.fold_num)

        models_for_ensemble = []

        for model_num in range(params.num_models):  # Number of models for the deep-ensemble
            print(f"\nWorking on {config_file} - model {model_num + 1}:")
            pretrained_subdir = "pretrained" if params.pretrained else "random"
            log_subdir = f"{params.model}/{pretrained_subdir}/{params.dataset_name}/model{model_num}"
            writer = SummaryWriter(log_dir=f"{log_dir}/{log_subdir}", comment=f"{log_subdir}")

            model = model_classes[params.model](params.dataset_name, params.pretrained).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
            # Only pretrained here
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Always pretrained here
            acc, best_acc, best_acc_epoch = train.train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn,
                                                     writer, params, params.fold_num, scheduler, model_num)

            utils.wrtie_to_csv(
                data=[params.model, model_num, params.dataset_name, params.pretrained, params.fold_num, acc, best_acc, best_acc_epoch],
                columns=columns, path=results_path)
            print(
                f"Saved results for {params.model} {model_num} | {params.dataset_name} | Pretrained- {params.pretrained} | Fold {params.fold_num}")

            models_for_ensemble.append(model)

        # Deep ensemble model part
        ensemble_model = DeepEnsemble(models_for_ensemble)
        acc = validate.evaluate(ensemble_model, device, val_loader)

        utils.wrtie_to_csv(
            data=[f"ensemble-{params.model}", "-", params.dataset_name, params.pretrained, params.fold_num, acc, "-", "-"],
            columns=columns, path=results_path)

        print(
            f"Saved results for ensemble-{params.model} | {params.dataset_name} | pretrained- {params.pretrained} | fold {params.fold_num}")
