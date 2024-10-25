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
    }
 
@torch.no_grad
def iou_from_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    intersection = (mask1 * mask2).sum(dim=1)
    area1, area2 = mask1.sum(dim=1), mask2.sum(dim=1)
    union = (area1 + area2) - intersection
    
    return torch.where(
        union == 0,
        torch.tensor(1., device=mask1.device),
        intersection / union,
    )
    