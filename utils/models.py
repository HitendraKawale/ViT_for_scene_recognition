"""Model and normalization helpers shared across the scripts.

A single Hugging Face ``AutoModelForImageClassification`` path means any
image-classification backbone on the Hub works just by changing the
``model_name`` in a config -- ViT, DINOv2, DeiT, BEiT, Swin, ConvNeXt, etc.
ResNet is kept on the torchvision path as a CNN baseline.
"""
import torch.nn as nn
from torchvision import models as tv_models
from transformers import AutoModelForImageClassification, AutoImageProcessor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# torchvision CNN baselines, mapped to their weight enums.
TORCHVISION_MODELS = {
    "resnet50": (tv_models.resnet50, "IMAGENET1K_V2"),
    "resnet18": (tv_models.resnet18, "IMAGENET1K_V1"),
}


def is_torchvision_model(model_name):
    return model_name in TORCHVISION_MODELS


def get_normalization(model_name):
    """Return (mean, std) appropriate for the given model."""
    if is_torchvision_model(model_name):
        return IMAGENET_MEAN, IMAGENET_STD
    extractor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    return extractor.image_mean, extractor.image_std


def build_model(model_name, num_classes, attn_implementation=None):
    """Build a classification model with a fresh `num_classes`-way head."""
    if is_torchvision_model(model_name):
        ctor, weights = TORCHVISION_MODELS[model_name]
        model = ctor(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    kwargs = {"num_labels": num_classes, "ignore_mismatched_sizes": True}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForImageClassification.from_pretrained(model_name, **kwargs)


def forward_logits(model, model_name, images):
    """Run a forward pass and return raw class logits for any backbone."""
    output = model(images)
    if is_torchvision_model(model_name):
        return output
    return output.logits
