import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa

from Project import Project

class ImgAugTransform:
    """
    Wrapper to allow imgaug to work with Pytorch transformation pipeline
    """

    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Affine(rotate=(-5, 5), mode='constant', cval=(0,0)),
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Sometimes(0.8, iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, )),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


val_transform = T.Compose([T.Resize((Project().input_width, Project().input_height)),
                           T.Grayscale(),
                           T.ToTensor()])

train_transform = T.Compose([T.Resize(((Project().input_width, Project().input_height))),
                             ImgAugTransform(),
                             T.ToPILImage(),
                             T.Grayscale(),
                             T.ToTensor()])
