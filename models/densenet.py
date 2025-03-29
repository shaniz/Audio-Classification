import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


class DenseNet(nn.Module):
    def __init__(self, dataset, pretrained=True):
        super(DenseNet, self).__init__()
        num_classes = 50 if dataset == "ESC" else 10
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class DenseNetWeightFusion(nn.Module):
    def __init__(self, dataset, layer, pretrained=True):
        super(DenseNetWeightFusion, self).__init__()
        num_classes = 50 if dataset == "ESC" else 10  # Only "ESC" here
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)

        layer_to_idx = {
            "conv0": 0,
            "denseblock1": 4,
            "denseblock2": 6,
            "denseblock3": 8,
            "denseblock4": 10
        }
        idx = layer_to_idx[layer]

        # Random weights for layers after idx layer, including idx layer
        for param in self.model.features[idx:].parameters():
            if param.ndimension() == 4:  # Convolutional layers
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif param.ndimension() == 2:  # Linear layers (e.g., classifier)
                init.xavier_uniform_(param)

    def forward(self, x):
        output = self.model(x)
        return output


class DenseNetWeightFreeze(nn.Module):
    def __init__(self, dataset, layer, pretrained=True):
        super(DenseNetWeightFreeze, self).__init__()
        num_classes = 50 if dataset == "ESC" else 10  # Only "ESC" here
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)

        layer_to_idx = {
            "conv0": 0,
            "denseblock1": 4,
            "denseblock2": 6,
            "denseblock3": 8,
            "denseblock4": 10
        }
        idx = layer_to_idx[layer]

        # freeze weights for layers before idx layer, including idx layer
        for param in self.model.features[:idx + 1].parameters():
            param.requires_grad = False  # default value is True

    def forward(self, x):
        output = self.model(x)
        return output


class DenseNetModelCutoff(nn.Module):
    def __init__(self, dataset, layer, pretrained=True):
        super(DenseNetModelCutoff, self).__init__()
        num_classes = 50 if dataset == "ESC" else 10  # Only "ESC" here
        self.model = models.densenet201(pretrained=pretrained)

        layer_to_param = {
            "denseblock3": {"idx": 8, "in_features": 256},
            "denseblock4": {"idx": 10, "in_features": 896}
        }
        idx = layer_to_param[layer]["idx"]

        # Remove layers
        self.model.features = nn.Sequential(*list(self.model.features.children())[:idx])
        self.model.classifier = nn.Linear(layer_to_param[layer]["in_features"], num_classes)

    def forward(self, x):
        output = self.model(x)
        return output
