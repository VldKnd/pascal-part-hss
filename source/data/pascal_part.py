import logging
import os
import zipfile

import gdown
import numpy
import PIL.Image
import torch.utils.data
import torchvision.transforms.functional

import source.constants

LOGGER = logging.getLogger(__name__)


class PascalPartDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = f"{source.constants.REPOSITORY_ROOT}/data",
        transform=None,
        target_transform=None,
        train: bool = True,
        download: bool = True,
    ):
        """
        Датасет Pascal-part.

        В папке `JPEGImages` находятся исходные изображения в формате jpeg.
        В папке `gt_masks` находятся маски сегментации в формате numpy.
        Загрузить маски можно при помощи функции `numpy.load()`.

        В датасете присутствуют 7 классов, обладающих следующей иерархической структурой (в скобках указан индекс класса):

        ├── (0) background
        └── body
            ├── upper_body
            |   ├── (1) low_hand
            |   ├── (6) up_hand
            |   ├── (2) torso
            |   └── (4) head
            └── lower_body
                ├── (3) low_leg
                └── (5) up_leg
        """
        self.check_dataset_consistancy_and_download(path_to_data_folder=root)
        self.root = root
        self.dataset_path = f"{self.root}/Pascal-part"
        self.path_to_masks = f"{self.dataset_path}/gt_masks"
        self.path_to_images = f"{self.dataset_path}/JPEGImages"
        self.class_to_name = {}
        self.name_to_class = {}
        self.download = download

        with open(f"{self.dataset_path}/classes.txt", "r") as file_with_classes:
            for line in file_with_classes.readlines():
                _class, name = line.split(":")
                _class, name = _class.strip(), name.strip()
                self.class_to_name[int(_class)] = name
                self.name_to_class[name] = int(_class)

        with open(
            (
                f"{self.dataset_path}/train_id.txt"
                if train
                else f"{self.dataset_path}/val_id.txt"
            ),
            "r",
        ) as file:
            self.file_ids = file.read().splitlines()

        self.transform = transform
        self.target_transform = target_transform

    def check_dataset_consistancy_and_download(self, path_to_data_folder: str):
        dataset_path = f"{path_to_data_folder}/Pascal-part"
        if not os.path.exists(dataset_path):
            raise RuntimeError(
                "Pascal-part dataset folder is not found.",
                "You should download it and put it in $ROOT/data folder",
            )

        if (
            not all(
                (
                    os.path.exists(f"{dataset_path}/gt_masks"),
                    os.path.exists(f"{dataset_path}/JPEGImages"),
                    os.path.exists(f"{dataset_path}/classes.txt"),
                    os.path.exists(f"{dataset_path}/train_id.txt"),
                    os.path.exists(f"{dataset_path}/val_id.txt"),
                )
            )
            and not self.download
        ):
            raise RuntimeError(
                "Pascal-part dataset is downloaded but not consistent.",
                "You should download it again and put it in $ROOT/data folder",
            )
        else:
            LOGGER.debug(
                "Dataset not found, downloading it into $REPOSITORY_ROOT/data folder."
            )
            os.mkdir(f"{source.constants.REPOSITORY_ROOT}/data")
            gdown.download(
                "https://drive.google.com/file/d/1unIkraozhmsFtkfneZVhw8JMOQ8jv78J",
                f"{source.constants.REPOSITORY_ROOT}/data/Pascal-part.zip",
            )
            LOGGER.debug("Dataset downloaded. Unzipping it.")
            with zipfile.ZipFile(
                f"{source.constants.REPOSITORY_ROOT}/data/Pascal-part.zip", "r"
            ) as archive:
                archive.extractall(
                    f"{source.constants.REPOSITORY_ROOT}/data/Pascal-part"
                )
            LOGGER.debug("Extracting finished.")

    def __getitem__(self, index):
        file_index = self.file_ids[index]
        image = PIL.Image.open(f"{self.path_to_images}/{file_index}.jpg")
        mask_as_numpy_array = numpy.load(f"{self.path_to_masks}/{file_index}.npy")
        image_as_tensor = torchvision.transforms.functional.pil_to_tensor(image)
        mask_as_tensor = torch.from_numpy(mask_as_numpy_array)

        if self.transform:
            image_as_tensor = self.transform(image_as_tensor)

        if self.target_transform:
            mask_as_tensor = self.target_transform(mask_as_tensor)

        return image_as_tensor, mask_as_tensor

    def __len__(self):
        return len(self.file_ids)


class AugmentedPascalPartDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = f"{source.constants.REPOSITORY_ROOT}/data",
        transform=None,
        target_transform=None,
        train: bool = True,
        download: bool = False,
    ):
        """
        Датасет Pascal-part со случайным зеркальным отражением и имзенением засветления изображений.

        В папке `JPEGImages` находятся исходные изображения в формате jpeg.
        В папке `gt_masks` находятся маски сегментации в формате numpy.
        Загрузить маски можно при помощи функции `numpy.load()`.

        В датасете присутствуют 7 классов, обладающих следующей иерархической структурой (в скобках указан индекс класса):

        ├── (0) background
        └── body
            ├── upper_body
            |   ├── (1) low_hand
            |   ├── (6) up_hand
            |   ├── (2) torso
            |   └── (4) head
            └── lower_body
                ├── (3) low_leg
                └── (5) up_leg
        """
        self.check_dataset_consistancy_and_download(path_to_data_folder=root)
        self.root = root
        self.dataset_path = f"{self.root}/Pascal-part"
        self.path_to_masks = f"{self.dataset_path}/gt_masks"
        self.path_to_images = f"{self.dataset_path}/JPEGImages"
        self.class_to_name = {}
        self.name_to_class = {}
        self.download = download

        with open(f"{self.dataset_path}/classes.txt", "r") as file_with_classes:
            for line in file_with_classes.readlines():
                _class, name = line.split(":")
                _class, name = _class.strip(), name.strip()
                self.class_to_name[int(_class)] = name
                self.name_to_class[name] = int(_class)

        with open(
            (
                f"{self.dataset_path}/train_id.txt"
                if train
                else f"{self.dataset_path}/val_id.txt"
            ),
            "r",
        ) as file:
            self.file_ids = file.read().splitlines()

        self.transform = transform
        self.target_transform = target_transform

    def check_dataset_consistancy_and_download(self, path_to_data_folder: str):
        dataset_path = f"{path_to_data_folder}/Pascal-part"
        if not os.path.exists(dataset_path):
            raise RuntimeError(
                "Pascal-part dataset folder is not found.",
                "You should download it and put it in $ROOT/data folder",
            )

        if (
            not all(
                (
                    os.path.exists(f"{dataset_path}/gt_masks"),
                    os.path.exists(f"{dataset_path}/JPEGImages"),
                    os.path.exists(f"{dataset_path}/classes.txt"),
                    os.path.exists(f"{dataset_path}/train_id.txt"),
                    os.path.exists(f"{dataset_path}/val_id.txt"),
                )
            )
            and not self.download
        ):
            raise RuntimeError(
                "Pascal-part dataset is downloaded but not consistent.",
                "You should download it again and put it in $ROOT/data folder",
            )
        else:
            LOGGER.debug(
                "Dataset not found, downloading it into $REPOSITORY_ROOT/data folder."
            )
            os.mkdir(f"{source.constants.REPOSITORY_ROOT}/data")
            gdown.download(
                "https://drive.google.com/file/d/1unIkraozhmsFtkfneZVhw8JMOQ8jv78J",
                f"{source.constants.REPOSITORY_ROOT}/data/Pascal-part.zip",
            )
            LOGGER.debug("Dataset downloaded. Unzipping it.")
            with zipfile.ZipFile(
                f"{source.constants.REPOSITORY_ROOT}/data/Pascal-part.zip", "r"
            ) as archive:
                archive.extractall(
                    f"{source.constants.REPOSITORY_ROOT}/data/Pascal-part"
                )
            LOGGER.debug("Extracting finished.")

    def __getitem__(self, index):
        file_index = self.file_ids[index]
        image = PIL.Image.open(f"{self.path_to_images}/{file_index}.jpg")
        mask_as_numpy_array = numpy.load(f"{self.path_to_masks}/{file_index}.npy")
        image_as_tensor = torchvision.transforms.functional.pil_to_tensor(image)
        mask_as_tensor = torch.from_numpy(mask_as_numpy_array)

        flip_flag = torch.rand(1).item()
        distortion_flag = torch.rand(1).item()
        grayscale_flag = torch.rand(1).item()

        if grayscale_flag > 0.95:
            image_as_tensor = torchvision.transforms.functional.rgb_to_grayscale(
                image_as_tensor, num_output_channels=3
            )

        if flip_flag > 0.5:
            image_as_tensor = torchvision.transforms.functional.hflip(image_as_tensor)
            mask_as_tensor = torchvision.transforms.functional.hflip(
                mask_as_tensor.unsqueeze(0)
            )

        if distortion_flag > 0.0 and distortion_flag < 0.25:
            image_as_tensor = torchvision.transforms.functional.equalize(
                image_as_tensor
            )
        elif distortion_flag > 0.25 and distortion_flag < 0.5:
            image_as_tensor = torchvision.transforms.functional.adjust_saturation(
                image_as_tensor, saturation_factor=0.5
            )
        elif distortion_flag > 0.5 and distortion_flag < 0.75:
            image_as_tensor = torchvision.transforms.functional.adjust_saturation(
                image_as_tensor, saturation_factor=2
            )

        if self.transform:
            image_as_tensor = self.transform(image_as_tensor)

        if self.target_transform:
            mask_as_tensor = self.target_transform(mask_as_tensor)

        return image_as_tensor, mask_as_tensor

    def __len__(self):
        return len(self.file_ids)
