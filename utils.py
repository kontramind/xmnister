import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from pathlib import Path
from skimage.io import imsave

import emnist

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.interactive(False)

def data_download() -> None:
    emnist.ensure_cached_data()


def _data_preprocess_test(dataset: str, data_dir: Path) -> None:
    usage = 'test'
    data_subdir = data_dir / usage
    data_subdir.mkdir(parents=True, exist_ok=True)

    images, labels = emnist.extract_samples(dataset=dataset, usage=usage)
    unique_labels = np.unique(labels)
    image_counters = { label:0 for label in unique_labels}
    for image, label in zip(images, labels):
        image = np.dstack((image,image,image))
        per_label_subdir = data_subdir/f"emnist_label_{label}"
        per_label_subdir.mkdir(parents=True, exist_ok=True)
        imsave(per_label_subdir/f"emnist_label_{label}_image_{image_counters[label]}.png", image)
        image_counters[label] += 1


def _data_preprocess_train(dataset: str, data_dir: Path) -> None:
    _, test_labels = emnist.extract_samples(dataset=dataset, usage="test")
    test_data_size = test_labels.size

    val_data_subdir = data_dir / "val"
    val_data_subdir.mkdir(parents=True, exist_ok=True)

    train_data_subdir = data_dir / "train"
    train_data_subdir.mkdir(parents=True, exist_ok=True)

    images, labels = emnist.extract_samples(dataset=dataset, usage="train")

    val_images = images[-test_data_size:, :, :]
    val_labels = labels[-test_data_size:]
    unique_labels = np.unique(labels)
    image_counters = { label:0 for label in unique_labels}
    for image, label in zip(val_images, val_labels):
        image = np.dstack((image,image,image))
        per_label_subdir = val_data_subdir/f"emnist_label_{label}"
        per_label_subdir.mkdir(parents=True, exist_ok=True)
        imsave(per_label_subdir/f"emnist_label_{label}_image_{image_counters[label]}.png", image)
        image_counters[label] += 1

    train_images = images[:-test_data_size, :, :]
    train_labels = labels[:-test_data_size]
    image_counters = { label:0 for label in unique_labels}
    for image, label in zip(train_images, train_labels):
        image = np.dstack((image,image,image))
        per_label_subdir = train_data_subdir/f"emnist_label_{label}"
        per_label_subdir.mkdir(parents=True, exist_ok=True)
        imsave(per_label_subdir/f"emnist_label_{label}_image_{image_counters[label]}.png", image)
        image_counters[label] += 1



def data_preprocess(data_dir: Path) -> None:
    dataset = "byclass"
    _data_preprocess_test(dataset, data_dir)
    _data_preprocess_train(dataset, data_dir)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
    

def show_dataset(dataset, n=6):
    imgs = [dataset[i][0] for i in range(n)]
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()


def show_dl(dl, n=6):
    batch = None
    for batch in dl:
        break
    imgs = batch[0][:n]
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()


# def _data_download() -> None:
#     project = Project()
#     data_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'

#     print(f"downloading {data_url:}")
#     data_request = requests.get(data_url, stream=True)

#     block_size = 1024 #1 Kb
#     total_size = int(data_request.headers.get('content-length', 0))
#     progress = tqdm(total=total_size, unit='iB', unit_scale=True)

#     with open(project.data_dir / "matlab.zip", "wb") as data_zip:
#         for data in data_request.iter_content(block_size):
#             progress.update(len(data))
#             data_zip.write(data)


# def _data_preprocess() -> None:
#     from pathlib import Path
#     from zipfile import ZipFile

#     project = Project()
#     data_file_prefix = "matlab"
#     data_file_name = "emnist-byclass.mat"
#     print(f"unzipping {data_file_name}")
#     with ZipFile(project.data_dir / f"{data_file_prefix}.zip", 'r') as data_zip:
#         data_zip.extract(f"{data_file_prefix}/{data_file_name}", project.data_dir)