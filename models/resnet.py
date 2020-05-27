from functools import partial

import torch.nn as nn
from torchvision.models import resnet18, resnet34

def weights_init(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)
        if model.bias is not None:
            nn.init.zeros_(model.bias)


def resnet_finetune(model, n_classes):
    """
    This function prepares resnet to be finetuned by:
    1) cut-off the last layer
    2) insert new last layer with the correct classes number
    """
    for param in model.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(512, n_classes)

    model.apply(weights_init)

    return model


# replace the resnet18 function
resnet18 = partial(resnet_finetune, resnet18(pretrained=False))


# replace the resnet34 function
resnet34 = partial(resnet_finetune, resnet34(pretrained=False))
