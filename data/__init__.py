import numpy as np
from .MyDataset import MyDataset
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets.folder import ImageFolder

def get_dataloaders(
        train_dir,
        val_dir,
        test_dir,
        train_transform=None,
        val_transform=None,
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    # create the datasets
    train_ds = ImageFolder(root=train_dir, transform=train_transform)
    val_ds = ImageFolder(root=val_dir, transform=val_transform)
    test_ds = ImageFolder(root=test_dir, transform=val_transform)
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)

    return train_dl, val_dl, test_dl
