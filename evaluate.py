from ctypes import ArgumentError
from typing import Any

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

_  = torch.manual_seed(0)

ALLOWED_DEVICES = ['cuda', 'cpu']
ALLOWED_MODELS = ['resnet50', 'resnet101', 'deeplabv3', 'segformer', 'resnet101_weighted_loss']

def parse_args() -> dict[str, Any]:
    return {
        device_to_use:'cuda',
        model_type:'resnet50',
    }

def validate_args(args_as_dict: dict[str, Any]):
    if args_as_dict.get('device_to_use', 'cpu') not in ALLOWED_DEVICES:
        raise ArgumentError(f"device_to_use parameter has to be one of {ALLOWED_DEVICES}")
    
    if args_as_dict.get('model_type', 'resnet101') not in ALLOWED_MODELS:
        raise ArgumentError(f"model_type parameter has to be one of {ALLOWED_MODELS}")
    
def get_model_and_transform(model_type: str = 'resnet101'):
    if model_type == 'resnet50':
        weight_name = 'augmented_resnet50_schedule_5000_resize_336_448'
        path_to_checkpoint = (
            f'{source.constants.REPOSITORY_ROOT}/' +
            'checkpoints/' + 
            weight_name
        )
        checkpoint = torch.load(path_to_checkpoint)
        model = torchvision.models.segmentation.fcn_resnet50()
        model.load_state_dict(checkpoint['model'])
        transforms = torchvision.transforms.Compose(
            [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((336, 448)),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Lambda(lambda image_as_int_tensor: image_as_int_tensor / 255.),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif model_type == 'resnet101':
        weight_name = 'augmented_resnet101_schedule_5000_resize_336_448'
        path_to_checkpoint = (
            f'{source.constants.REPOSITORY_ROOT}/' +
            'checkpoints/' + 
            weight_name
        )
        checkpoint = torch.load(path_to_checkpoint)
        model = torchvision.models.segmentation.fcn_resnet101()
        model.load_state_dict(checkpoint['model'])
        transforms = torchvision.transforms.Compose(
            [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((336, 448)),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Lambda(lambda image_as_int_tensor: image_as_int_tensor / 255.),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    elif model_type == 'deeplabv3':
        weight_name = 'augmented_deeplabv3_schedule_5000_resize_336_448'
        path_to_checkpoint = (
            f'{source.constants.REPOSITORY_ROOT}/' +
            'checkpoints/' + 
            weight_name
        )
        checkpoint = torch.load(path_to_checkpoint)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
        model.load_state_dict(checkpoint['model'])
        transforms = torchvision.transforms.Compose(
            [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((336, 448)),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Lambda(lambda image_as_int_tensor: image_as_int_tensor / 255.),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    elif model_type == 'segformer':
        weight_name = 'augmented_segformer_schedule_5000_resize_336_448'
        path_to_checkpoint = (
            f'{source.constants.REPOSITORY_ROOT}/' +
            'checkpoints/' + 
            weight_name
        )
        checkpoint = torch.load(path_to_checkpoint)
        model = transformers.SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )
        
        model.load_state_dict(checkpoint['model'])

        transforms = torchvision.transforms.Compose(
            [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((336, 448)),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Lambda(lambda image_as_int_tensor: image_as_int_tensor / 255.),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    elif model_type == 'resnet101_weighted_loss':

    else:
        raise ArgumentError(f"model_type parameter has to be one of {ALLOWED_MODELS}")

    return model, transforms
    
if __name__ == '__main__':
    args_as_dict = parse_args()
    validate_args(args_as_dict)

    device_to_use = args_as_dict['device_to_use']
    model_type = args_as_dict['model_type']

    cpu_device = torch.device('cpu')
    device = torch.device(device_to_use)

    with torch.no_grad():
        
        if model_type == 'resnet50':
            weight_name = 'augmented_resnet50_schedule_5000_resize_336_448'
            path_to_checkpoint = (
                f'{source.constants.REPOSITORY_ROOT}/' +
                'checkpoints/' + 
                weight_name
            )
            checkpoint = torch.load(path_to_checkpoint)
            model = torchvision.models.segmentation.fcn_resnet50()
            model.load_state_dict(checkpoint['model'])

            image_transform = image_transform = torchvision.transforms.Compose(
                [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((336, 448)),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Lambda(lambda image_as_int_tensor: image_as_int_tensor / 255.),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            evaluation_dataset = source.data.PascalPartDataset(
                transform=image_transform,
                train=False,
            )
            _ = model.eval()
        else:
            raise RuntimeError()
        
        
        evaluation_metrics_tracker = {index:[] for index in range(0, 10)}
        
        for image, mask in tqdm.tqdm(evaluation_dataset):
            image = image.to(device)
            mask = mask.to(device)
            
            mask_shape = mask.shape
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)

            output_logits = model(image).logits

            log_probabilities = torch.nn.functional.log_softmax(
                output_logits,
                dim=1, # 1 x C x H x W
            )

            upsampled_log_probabilities = torch.nn.functional.interpolate(
                input=log_probabilities,
                size=mask_shape,
            )

            intersection_over_union_per_class = source.utils.log_probabilities_and_mask_iou_per_class(
                log_probabilities=upsampled_log_probabilities,
                masks=mask,
                classes=list(evaluation_dataset.class_to_name.keys()),
            )

            for object_class in range(0, 10):
                filtered_iou = intersection_over_union_per_class[object_class][
                    intersection_over_union_per_class[object_class] >= 0.
                ].to(cpu_device).detach()
                if filtered_iou.numel():
                    evaluation_metrics_tracker[object_class].append(filtered_iou.mean().item())

        print(
            "IoU - ",
            *[
                f"{evaluation_dataset.class_to_name[object_class]} : " +
                f"{sum(evaluation_metrics_tracker[object_class]) / len(evaluation_metrics_tracker[object_class]):.3f} \t "
                for object_class in list(evaluation_dataset.class_to_name.keys())
            ],
            f"upper_body: {sum(evaluation_metrics_tracker[7]) / len(evaluation_metrics_tracker[7]):.3f} \t ",
            f"lower_body: {sum(evaluation_metrics_tracker[8]) / len(evaluation_metrics_tracker[8]):.3f} \t ",
            f"body: {sum(evaluation_metrics_tracker[9]) / len(evaluation_metrics_tracker[9]):.3f} \t ",
        )