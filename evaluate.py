import argparse
import logging
import os
from ctypes import ArgumentError
from typing import Any

import gdown
import torch
import torch.hub
import torch.optim
import torch.utils.data
import torchvision.models.segmentation
import torchvision.transforms
import tqdm
import transformers

import source.constants
import source.data
import source.modules
import source.utils

_ = torch.manual_seed(0)


LOGGER = logging.getLogger(__name__)
PATH_TO_CHECKPOINTS = f"{source.constants.REPOSITORY_ROOT}/checkpoints"
PATH_TO_DATA = f"{source.constants.REPOSITORY_ROOT}/data"

ALLOWED_DEVICES = {"cuda", "cpu"}

ALLOWED_MODELS = {"resnet50", "resnet101", "deeplabv3", "segformer", "resnet101_long"}

MODEL_TO_WEIGHTS_LINKS = {
    "resnet50": "https://drive.google.com/file/d/16mAgAtS8qdks_XYzuCRAjF7vcBEpWi1H",
    "resnet101": "https://drive.google.com/file/d/1nt1IRujzH_dowIUMQudwopzgBL708nkO",
    "deeplabv3": "https://drive.google.com/file/d/1r2i0tAIMzKB0sgNJcAG5zCn6ZMLtKA1s",
    "segformer": "https://drive.google.com/file/d/1ayjtJxx3zDnTJ97ipwdFzV5YKGwvtkX9",
    "resnet101_long": "https://drive.google.com/file/d/19f1S7_YUEUQ3LRERXj85S7tjPeoeouGE",
}


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Evaluating pretrained model on test part of Pascal-parts"
    )
    parser.add_argument(
        "--device_to_use",
        choices=["cpu", "cuda"],
        type=str,
        help="On which device to run inference.",
        default="cpu",
    )

    parser.add_argument(
        "--model_type",
        choices=["resnet50", "resnet101", "deeplabv3", "segformer", "resnet101_long"],
        type=str,
        help="Which pre-trained model to use.",
        default="reset50",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Wether to show additional information or not.",
    )

    args = parser.parse_args()

    return args.__dict__


def validate_args(args_as_dict: dict[str, Any]):
    if args_as_dict.get("device_to_use", "cpu") not in ALLOWED_DEVICES:
        raise ArgumentError(
            f"device_to_use parameter has to be one of {ALLOWED_DEVICES}"
        )

    if args_as_dict.get("model_type", "resnet101") not in ALLOWED_MODELS:
        raise ArgumentError(f"model_type parameter has to be one of {ALLOWED_MODELS}")


def check_and_download_models_weights(model_type: str = "resnet101"):
    if model_type not in MODEL_TO_WEIGHTS_LINKS:
        raise ArgumentError(f"model_type parameter has to be one of {ALLOWED_MODELS}")

    path_to_model_weights = f"{PATH_TO_CHECKPOINTS}/{model_type}.pth"
    if not os.path.exists(path_to_model_weights):
        LOGGER.debug(f"Downloading {model_type} weights from gdrive.")
        gdown.download(MODEL_TO_WEIGHTS_LINKS[model_type], path_to_model_weights)
        LOGGER.debug(f"Weights succesfuly downloaded.")

    return path_to_model_weights


def get_model_and_transform(model_type: str = "resnet101"):
    path_to_checkpoint = (
        f"{source.constants.REPOSITORY_ROOT}/" "checkpoints/" f"{model_type}.pth"
    )
    model_weights = torch.load(path_to_checkpoint)["model"]

    if model_type == "resnet50":
        model = torchvision.models.segmentation.fcn_resnet50()
        model.load_state_dict(model_weights)
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((336, 448)),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Lambda(
                    lambda image_as_int_tensor: image_as_int_tensor / 255.0
                ),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    elif model_type == "resnet101":
        model = torchvision.models.segmentation.fcn_resnet101()
        model.load_state_dict(model_weights)
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((336, 448)),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Lambda(
                    lambda image_as_int_tensor: image_as_int_tensor / 255.0
                ),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    elif model_type == "deeplabv3":
        model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50")
        model.load_state_dict(model_weights)
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((336, 448)),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Lambda(
                    lambda image_as_int_tensor: image_as_int_tensor / 255.0
                ),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    elif model_type == "segformer":
        model = transformers.SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )
        model.load_state_dict(model_weights)

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((336, 448)),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Lambda(
                    lambda image_as_int_tensor: image_as_int_tensor / 255.0
                ),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    elif model_type == "resnet101_long":
        model = torchvision.models.segmentation.fcn_resnet101()
        model.load_state_dict(model_weights)
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((336, 448)),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Lambda(
                    lambda image_as_int_tensor: image_as_int_tensor / 255.0
                ),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    else:
        raise ArgumentError(f"model_type parameter has to be one of {ALLOWED_MODELS}")

    return model, transforms


def get_model_logits(
    image_batch: torch.Tensor, model: torch.nn.Sequential
) -> torch.Tensor:
    if model_type == "resnet50":
        logits = model(image_batch)

    elif model_type == "resnet101":
        logits = model(image_batch)

    elif model_type == "deeplabv3":
        logits = model(image_batch)["out"]

    elif model_type == "segformer":
        logits = model(image_batch).logits

    elif model_type == "resnet101_long":
        logits = model(image_batch)

    else:
        raise ArgumentError(f"model_type parameter has to be one of {ALLOWED_MODELS}")

    return logits


if __name__ == "__main__":
    args_as_dict = parse_args()
    validate_args(args_as_dict)

    device_to_use = args_as_dict["device_to_use"]
    model_type = args_as_dict["model_type"]
    verbose = args_as_dict["verbose"]

    logger_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logger_level,
    )
    LOGGER.setLevel(logger_level)

    cpu_device = torch.device("cpu")
    device = torch.device(device_to_use)

    check_and_download_models_weights(model_type=model_type)

    with torch.no_grad():

        model, image_transforms = get_model_and_transform(model_type=model_type)

        evaluation_dataset = source.data.PascalPartDataset(
            transform=image_transforms,
            train=False,
        )
        _ = model.eval()

        evaluation_metrics_tracker = {index: [] for index in range(0, 10)}

        for image, mask in tqdm.tqdm(evaluation_dataset):
            image = image.to(device)
            mask = mask.to(device)

            mask_shape = mask.shape
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)

            output_logits = get_model_logits(image, model).logits

            log_probabilities = torch.nn.functional.log_softmax(
                output_logits,
                dim=1,  # 1 x C x H x W
            )

            upsampled_log_probabilities = torch.nn.functional.interpolate(
                input=log_probabilities,
                size=mask_shape,
            )

            intersection_over_union_per_class = (
                source.utils.log_probabilities_and_mask_iou_per_class(
                    log_probabilities=upsampled_log_probabilities,
                    masks=mask,
                    classes=list(evaluation_dataset.class_to_name.keys()),
                )
            )

            for object_class in range(0, 10):
                filtered_iou = (
                    intersection_over_union_per_class[object_class][
                        intersection_over_union_per_class[object_class] >= 0.0
                    ]
                    .to(cpu_device)
                    .detach()
                )
                if filtered_iou.numel():
                    evaluation_metrics_tracker[object_class].append(
                        filtered_iou.mean().item()
                    )

        LOGGER.info(
            "IoU - ",
            *[
                f"{evaluation_dataset.class_to_name[object_class]} : "
                + f"{sum(evaluation_metrics_tracker[object_class]) / len(evaluation_metrics_tracker[object_class]):.3f} \t "
                for object_class in list(evaluation_dataset.class_to_name.keys())
            ],
            f"upper_body: {sum(evaluation_metrics_tracker[7]) / len(evaluation_metrics_tracker[7]):.3f} \t ",
            f"lower_body: {sum(evaluation_metrics_tracker[8]) / len(evaluation_metrics_tracker[8]):.3f} \t ",
            f"body: {sum(evaluation_metrics_tracker[9]) / len(evaluation_metrics_tracker[9]):.3f} \t ",
        )
