import torch
import torch.nn as nn
import utils
import models
import pandas as pd
import dataloaders.datasetaug
import dataloaders.datasetnormal
import train
from tensorboardX import SummaryWriter
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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_files = utils.list_files("configC/resnet/pretrained/esc.json")
    results_path = "resultsC.csv"

    # For saving results in csv later
    columns = ["Model", "Model Number", "Dataset", "Pretrained", "Fold", "Accuracy", "Best Accuracy"]
    pd.DataFrame(columns=columns).to_csv(results_path, index=False)
    fold_num = 1  # Working on this fold always
    num_models = 1  # Number of models for the deep-ensemble

    model_classes = {
        "densenet": models.densenet.DenseNet,
        "resnet": models.resnet.ResNet,
        "inception": models.inception.Inception,
    }

    for config_file in config_files:
        params = utils.Params(config_file)

        train_loader = dataloaders.datasetaug.fetch_dataloader(
            "{}training128mel{}.pkl".format(params.data_dir, fold_num), params.dataset_name, params.batch_size,
            params.num_workers, 'train', params.model)
        val_loader = dataloaders.datasetaug.fetch_dataloader(
            "{}validation128mel{}.pkl".format(params.data_dir, fold_num), params.dataset_name, params.batch_size,
            params.num_workers, 'validation', params.model)

        loss_fn = nn.CrossEntropyLoss()
        models_for_ensemble = []

        for model_num in range(num_models):
            print(f"Working on {config_file} - model {model_num + 1}:")

            writer = SummaryWriter(log_dir=f"runs/{params.model}/{params.pretrained}/{params.dataset_name}/{model_num}",
                                   comment=f"{params.model}/{params.pretrained}/{params.dataset_name}/ensemble/model{model_num}")

            model = model_classes[params.model](params.dataset_name, params.pretrained).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Always pretrained here
            acc, best_acc = train.train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn,
                                                     writer, params, fold_num, scheduler, model_num)

            # Writing results to csv
            df = pd.DataFrame(
                data=[[params.model, model_num, params.dataset_name, params.pretrained, fold_num, acc, best_acc]],
                columns=columns)
            df.to_csv(results_path, mode='a', header=False, index=False)
            print(
                f"Saved results for {params.model} {model_num} | {params.dataset_name} | Pretrained- {params.pretrained} | Fold {fold_num}")

            models_for_ensemble.append(model)

        # Creating the deep ensemble model
        ensemble_model = DeepEnsemble(models_for_ensemble)
        # Calculating validation accuracy
        acc = validate.evaluate(ensemble_model, device, val_loader)
        # Writing results to csv
        df = pd.DataFrame(
            data=[[f"ensemble-{params.model}", "-", params.dataset_name, params.pretrained, fold_num, acc, "-"]],
            columns=columns)
        df.to_csv(results_path, mode='a', header=False, index=False)
        print(
            f"Saved results for ensemble-{params.model} | {params.dataset_name} | pretrained- {params.pretrained} | model {params.model} | fold {fold_num}")
