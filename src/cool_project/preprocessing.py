from torchvision import transforms
from typing import List

# ---- Atomic building blocks ----

BASE_TRANSFORMS = {
    "to_tensor": transforms.ToTensor(),
    "normalize_imagenet": transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet means
        std=[0.229, 0.224, 0.225],    # ImageNet stds
    ),
    "hflip": transforms.RandomHorizontalFlip(),
    "vflip": transforms.RandomVerticalFlip(),
    "color_jitter_light": transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
    "color_jitter_strong": transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
    ),
    "random_resized_crop_224": transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
}


def make_resize(size: int):
    """
    Resize to (size, size) using Bicubic interpolation (good for medical images).
    """
    return transforms.Resize(
        (size, size),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )


def build_transform(transform_names: List[str]):
    """
    Generic builder from a list of keys in BASE_TRANSFORMS.
    """
    steps = []
    for name in transform_names:
        if name == "resize_dynamic":
            raise ValueError("Use build_pipeline_for_model(size, ...) for dynamic resize.")
        steps.append(BASE_TRANSFORMS[name])
    return transforms.Compose(steps)


# ---- Model-aware pipelines ----

def build_pipeline_for_model(
    model_type: str,
    size: int,
    mode: str = "train",
):
    """
    model_type: "cnn", "vit", "dino", etc.
    size: target input size (e.g., 224, 512, ...)
    mode: "train" or "eval"

    Returns a torchvision.transforms.Compose.
    """
    resize = make_resize(size)

    if model_type == "cnn":
        # Typical ImageNet CNN pipeline
        if mode == "train":
            return transforms.Compose([
                resize,
                BASE_TRANSFORMS["hflip"],
                BASE_TRANSFORMS["color_jitter_light"],
                BASE_TRANSFORMS["to_tensor"],
                BASE_TRANSFORMS["normalize_imagenet"],
            ])
        else:
            return transforms.Compose([
                resize,
                BASE_TRANSFORMS["to_tensor"],
                BASE_TRANSFORMS["normalize_imagenet"],
            ])

    elif model_type == "vit":
        # ViT-style: often uses bicubic and simple augmentations
        if mode == "train":
            return transforms.Compose([
                resize,
                BASE_TRANSFORMS["to_tensor"],
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])
        else:
            return transforms.Compose([
                resize,
                BASE_TRANSFORMS["to_tensor"],
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])

    elif model_type == "dino":
        # Stronger augmentations for self-supervised / DINO-like
        if mode == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                BASE_TRANSFORMS["to_tensor"],
                BASE_TRANSFORMS["normalize_imagenet"],
            ])
        else:
            return transforms.Compose([
                resize,
                BASE_TRANSFORMS["to_tensor"],
                BASE_TRANSFORMS["normalize_imagenet"],
            ])

    else:
        # Default "medical" pipeline (like we discussed before)
        if mode == "train":
            return transforms.Compose([
                resize,
                BASE_TRANSFORMS["hflip"],
                BASE_TRANSFORMS["color_jitter_light"],
                BASE_TRANSFORMS["to_tensor"],
                BASE_TRANSFORMS["normalize_imagenet"],
            ])
        else:
            return transforms.Compose([
                resize,
                BASE_TRANSFORMS["to_tensor"],
                BASE_TRANSFORMS["normalize_imagenet"],
            ])


def build_transformation_pipeline(size: int, train: bool = True):
    """
    Wrapper so existing code that calls build_medical_pipeline still works.
    Uses ViT-style transforms by default.
    """
    mode = "train" if train else "eval"
    return build_pipeline_for_model("vit", size, mode)

'''
# Single building blocks
TRANSFORMS = {
    "resize_224": transforms.Resize((224, 224)),
    "resize_256": transforms.Resize((256, 256)),
    "center_crop_224": transforms.CenterCrop(224),
    "random_crop_224": transforms.RandomCrop(224),
    "hflip": transforms.RandomHorizontalFlip(),
    "vflip": transforms.RandomVerticalFlip(),
    "color_jitter": transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ),
    "to_tensor": transforms.ToTensor(),
    "normalize_imagenet": transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
}

def build_transform(transform_names):
    """
    Build a torchvision transform pipeline from a list of names.
    """
    steps = [TRANSFORMS[name] for name in transform_names]
    return transforms.Compose(steps)

'''
'''
TRANSFORMS = {
    "to_tensor": transforms.ToTensor(),
    "normalize_imagenet": transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    "hflip": transforms.RandomHorizontalFlip(),
    "color_jitter": transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ),
}

def make_resize(size):
    # high-quality interpolation
    return transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)

def build_medical_pipeline(size):
    return transforms.Compose([
        make_resize(size),
        TRANSFORMS["hflip"],
        TRANSFORMS["color_jitter"],
        TRANSFORMS["to_tensor"],
        TRANSFORMS["normalize_imagenet"],
    ])


'''