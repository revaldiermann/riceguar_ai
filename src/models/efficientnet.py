
import timm
import torch.nn as nn


MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 3


class RiceGuardModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=NUM_CLASSES,
        )

    def forward(self, x):
        return self.model(x)


def build_model(pretrained=True):
    return RiceGuardModel(pretrained=pretrained)
