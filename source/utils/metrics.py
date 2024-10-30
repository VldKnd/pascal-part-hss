import torch
import torch.nn.functional


@torch.no_grad
def log_probabilities_and_mask_iou_per_class(
    log_probabilities: torch.Tensor,
    masks: torch.Tensor,
    classes: list[int]
) -> dict[int, torch.Tensor]:
    mask_predictions = torch.argmax(log_probabilities, dim=1)
    return {
        objects_class: iou_from_masks(
            mask_predictions == objects_class,
            masks == objects_class,
        ) for objects_class in classes
    } | {
        7: iou_from_masks(
            (mask_predictions == 1) | (mask_predictions == 6) |
            (mask_predictions == 2) | (mask_predictions == 4) ,
            (masks == 1) | (masks == 6) &
            (masks == 2) | (masks == 4) ,
        ), # upper_body
        8: iou_from_masks(
            (mask_predictions == 3) | (mask_predictions == 5),
            (masks == 3) | (masks == 5),
        ), # lower_body
        9: iou_from_masks(
            mask_predictions > 0,
            masks > 0,
        ), # body
    }
 
@torch.no_grad
def iou_from_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    intersection = (mask1 * mask2).sum(dim=(1, 2))
    area1, area2 = mask1.sum(dim=(1, 2)), mask2.sum(dim=(1, 2))
    union = (area1 + area2) - intersection
    
    return torch.where(
        union == 0,
        torch.tensor(-1., device=mask1.device),
        intersection / union,
    )
    