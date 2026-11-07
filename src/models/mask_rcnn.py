"""Mask R-CNN model construction."""

import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(num_classes: int = 7, pretrained: bool = True) -> torch.nn.Module:
    """Create a COCO-pretrained Mask R-CNN with CarDD output heads."""
    if pretrained:
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    else:
        model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)

    # Replace the COCO box classifier with a CarDD classifier.
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)

    # Replace the COCO mask classifier with a CarDD mask classifier.
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model
