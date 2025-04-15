import torch.nn as nn
import torchvision.models as models


class EfficientNet(nn.Module):
    def __init__(self, dataset, pretrained=True, version='b0'):
        super(EfficientNet, self).__init__()
        num_classes = 50 if dataset == "ESC" else 10

        model_func = getattr(models, f"efficientnet_{version}")
        self.model = model_func(pretrained=pretrained)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
