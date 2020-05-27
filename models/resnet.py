from functools import partial

import torch.nn as nn
from torchvision.models import resnet18, resnet34


def resnet_finetune(model, n_classes):
    """
    This function prepares resnet to be finetuned by:
    1) cut-off the last layer
    2) replace the last layer with a new one with the correct classes number
    """
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(512, n_classes)
    return model


# replace the resnet18 function
resnet18 = partial(resnet_finetune, resnet18(pretrained=True))


# replace the resnet34 function
resnet34 = partial(resnet_finetune, resnet34(pretrained=True))
