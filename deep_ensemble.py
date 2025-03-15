import torch
import torch.nn as nn
import utils
import models
import pandas as pd
import dataloaders.datasetaug
import dataloaders.datasetnormal
import train
from tensorboardX import SummaryWriter


# class Model(nn.Module):
#     def __init__(self, model, dataset_name, checkpoint_path, pretrained):
#         super(Model, self).__init__()
#
#         if model == "densenet":
#             self.model = models.densenet.DenseNet(dataset_name, pretrained).to(device)
#         elif model == "resnet":
#             self.model = models.resnet.ResNet(dataset_name, pretrained).to(device)
#         elif model == "inception":
#             self.model = models.inception.Inception(dataset_name, pretrained).to(device)
#
#         utils.load_checkpoint(checkpoint=checkpoint_path, model=self.model)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class DeepEnsemble(nn.Module):
#     def __init__(self, model, dataset_name, checkpoint_dir, num_models=5, pretrained=True):
#         super(DeepEnsemble, self).__init__()
#         self.models = nn.ModuleList([
#             Model(model=model, dataset_name=dataset_name, checkpoint_path=f"{checkpoint_dir}/last{i+1}.pth.tar", pretrained=pretrained)
#              for i in range(num_models)])
#
#     def forward(self, x):
#         outputs = torch.stack([model(x) for model in self.models],
#                               dim=0)  # Shape: (num_models, batch_size, num_classes)
#         ensemble_output = torch.mean(outputs, dim=0)  # Averaging predictions
#         return ensemble_output


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ensemble_model = DeepEnsemble("resnet", "etc", "checkpointsB1/resnet/pretrained/esc").to(device)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_files = utils.list_files("../configC/resnet/pretrained/esc.json")
    results_path = "resultsC.csv"
    print(config_files)

    # For saving results in csv later
    columns = ["Model", "Model Number", "Dataset", "Pretrained", "Fold", "Accuracy", "Best Accuracy"]
    pd.DataFrame(columns=columns).to_csv(results_path, index=False)
    fold = 1
    num_models = 1

    for config_file in config_files:
        print(config_file)
        params = utils.Params(config_file)

        if params.dataaug:
            train_loader = dataloaders.datasetaug.fetch_dataloader("{}training128mel{}.pkl".format(params.data_dir, fold), params.dataset_name, params.batch_size, params.num_workers, 'train', params.model)
            val_loader = dataloaders.datasetaug.fetch_dataloader("{}validation128mel{}.pkl".format(params.data_dir, fold), params.dataset_name, params.batch_size, params.num_workers, 'validation', params.model)
        else:
            train_loader = dataloaders.datasetnormal.fetch_dataloader("{}training128mel{}.pkl".format(params.data_dir, fold), params.dataset_name, params.batch_size, params.num_workers, params.model)
            val_loader = dataloaders.datasetnormal.fetch_dataloader("{}validation128mel{}.pkl".format(params.data_dir, fold), params.dataset_name, params.batch_size, params.num_workers, params.model)

        writer = SummaryWriter(comment=params.dataset_name)
        if params.model == "densenet":
            model = models.densenet.DenseNet(params.dataset_name, params.pretrained).to(device)
        elif params.model == "resnet":
            model = models.resnet.ResNet(params.dataset_name, params.pretrained).to(device)
        elif params.model == "inception":
            model = models.inception.Inception(params.dataset_name, params.pretrained).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        if params.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            scheduler = None

        for model_num in range(num_models):
            print(f"Working on {config_file}-{model_num + 1}:")

            acc, best_acc = train.train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, fold, scheduler, model_num)

            # Writing results to csv
            df = pd.DataFrame(data=[[params.model, model_num, params.dataset_name, params.pretrained, fold, acc, best_acc]],
                              columns=columns)
            df.to_csv(results_path, mode='a', header=False, index=False)
            print(f"Saved results for {params.model} number {model_num} | {params.dataset_name} | Pretrained- {params.pretrained} | Fold {fold}")
