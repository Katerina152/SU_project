from torchvision import transforms
from typing import List
from torchvision.transforms import functional as F
import numpy as np
import torch


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

def make_resize_square(size: int):
    # your current behavior (warps aspect ratio)
    return transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)

def make_resize_center_crop(size: int, crop_pct: float = 1.0):
    # preserve aspect ratio, then center crop
    resize_size = int(round(size / crop_pct))
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
    ])


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
    backbone_type: str | None = None,
    model_name: str | None = None,
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

    elif model_type == "vit" :
        if backbone_type == "timm":
            geom = make_resize_center_crop(size, crop_pct=1.0)  # or 0.95 if you want that policy
            # ViT-style: often uses bicubic and simple augmentations
            if mode == "train":
                return transforms.Compose([
                    #transforms.RandomResizedCrop(
                        #size=size,
                        #scale=(0.6, 1.0),
                        #interpolation=transforms.InterpolationMode.BICUBIC,
                    #),
                    geom ,
                    BASE_TRANSFORMS["to_tensor"],
                    BASE_TRANSFORMS["normalize_imagenet"],
                ])
                #return transforms.Compose([
                    #resize,
                    # (iii) break pixel-level correspondence
                    #transforms.RandomRotation(15),      
                    #BASE_TRANSFORMS["hflip"],           
                    #BASE_TRANSFORMS["color_jitter_light"],  
                    #transforms.RandomResizedCrop(size=size, scale=(0.6, 1.0)),
                    #BASE_TRANSFORMS["to_tensor"],
                    #transforms.Normalize(
                        #mean=[0.5, 0.5, 0.5],
                        #std=[0.5, 0.5, 0.5],
                    #),
                #])
            else:
                #crop_pct = 0.95
                #resize_size = int(size / crop_pct)  # e.g., 224/0.95 ~= 236
                return transforms.Compose([
                    #transforms.Resize(
                        #resize_size,
                        #interpolation=transforms.InterpolationMode.BICUBIC,
                    #),
                    #transforms.CenterCrop(size),
                    geom,
                    BASE_TRANSFORMS["to_tensor"],
                    BASE_TRANSFORMS["normalize_imagenet"],
                
                ])
        
        elif backbone_type == "hf":
            if mode == "train":
                geom = make_resize_center_crop(size, crop_pct=1.0)  # or 0.95 if you want that policy
                return transforms.Compose([
                    geom,
                    BASE_TRANSFORMS["to_tensor"],
                    BASE_TRANSFORMS["normalize_imagenet"],
                ])
            else:
                geom = make_resize_center_crop(size, crop_pct=1.0)  # or 0.95 if you want that policy
                return transforms.Compose([
                    geom,
                    BASE_TRANSFORMS["to_tensor"],
                    BASE_TRANSFORMS["normalize_imagenet"],
                ])


    #elif model_type in ["dino", "dino_timm"]:
        # Stronger augmentations for self-supervised / DINO-like
        #if mode == "train":
            #return transforms.Compose([
                #resize,
                #transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                #BASE_TRANSFORMS["to_tensor"],
                #BASE_TRANSFORMS["normalize_imagenet"],
            #])
        #else:
            #return transforms.Compose([
                #resize,
                #BASE_TRANSFORMS["to_tensor"],
                #BASE_TRANSFORMS["normalize_imagenet"],
            #])

    else: 
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

'''
def build_aug_pipeline_for_model(
    model_type: str,
    size: int,
    mode: str = "train",
    backbone_type: str | None = None,
    model_name: str | None = None,
):
    """
    model_type: "cnn", "vit", "dino", etc.
    size: target input size (e.g., 224, 512, ...)
    mode: "train" or "eval"

    Returns a torchvision.transforms.Compose.
    """
    resize = make_resize(size)

    # only implement what you want; no fallback
    if not (model_type == "vit" and backbone_type == "timm"):
        raise ValueError(f"Aug pipeline only for vit+timm. Got model_type={model_type}, backbone_type={backbone_type}")

    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size=size,
                scale=(0.6, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomRotation(15),
            BASE_TRANSFORMS["to_tensor"],
            BASE_TRANSFORMS["normalize_imagenet"],
        ])
    else:
        return transforms.Compose([
            resize,
            BASE_TRANSFORMS["to_tensor"],
            BASE_TRANSFORMS["normalize_imagenet"],
        ])
'''

def build_aug_pipeline_for_model(
    model_type: str,
    size: int,
    mode: str = "train",
    backbone_type: str | None = None,
    model_name: str | None = None,
    aug_id: int = 0,  # 0..4
):
    resize = make_resize(size)

    # only implement what you want; no fallback
    if not (model_type == "vit" and backbone_type == "timm"):
        raise ValueError(
            f"Aug pipeline only for vit+timm. Got model_type={model_type}, backbone_type={backbone_type}"
        )

    # val/test: deterministic (no augmentation)
    if mode != "train":
        return transforms.Compose([
            resize,
            BASE_TRANSFORMS["to_tensor"],
            BASE_TRANSFORMS["normalize_imagenet"],
        ])

    # --- Shared: keep output size consistent ---
    # We'll use RandomResizedCrop in most views so every aug stays [size,size].
    # Blur should also happen on a resized/cropped image.
    if aug_id == 0:
        # BLUR view (1 blur)
        ops = [
            transforms.RandomResizedCrop(
                size=size, scale=(0.6, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        ]

    elif aug_id == 1:
        # ROTATION #1 (mild)
        ops = [
            transforms.RandomResizedCrop(
                size=size, scale=(0.6, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomRotation(10),
        ]

    elif aug_id == 2:
        # ROTATION #2 (strong)
        ops = [
            transforms.RandomResizedCrop(
                size=size, scale=(0.6, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomRotation(25),
        ]

    elif aug_id == 3:
        # CROP #1 (mild crop)
        ops = [
            transforms.RandomResizedCrop(
                size=size, scale=(0.6, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]

    elif aug_id == 4:
        # CROP #2 (strong crop)
        ops = [
            transforms.RandomResizedCrop(
                size=size, scale=(0.3, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]

    else:
        raise ValueError(f"aug_id must be in [0..4], got {aug_id}")

    return transforms.Compose([
        *ops,
        BASE_TRANSFORMS["to_tensor"],
        BASE_TRANSFORMS["normalize_imagenet"],
    ])


class SegmentationTransform:
    def __init__(self, size: int, train: bool = True):
        self.size = size
        self.train = train

    def __call__(self, img, mask):
        # 1) resize both
        img = F.resize(img, (self.size, self.size),
                       interpolation=F.InterpolationMode.BICUBIC)
        mask = F.resize(mask, (self.size, self.size),
                        interpolation=F.InterpolationMode.NEAREST)

        # 2) random flip with same decision
        if self.train and torch.rand(1) < 0.5:
            img  = F.hflip(img)
            mask = F.hflip(mask)

        # 3) image: to tensor + normalize
        img = F.to_tensor(img)
        img = F.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # 4) mask: to LongTensor of class ids
        mask_np = np.array(mask)
        mask_t  = torch.from_numpy(mask_np).long()

        return img, mask_t

def build_transformation_pipeline(size: int, train: bool = True, model_type: str = "vit", backbone_type=None, model_name=None):
    mode = "train" if train else "test"
    return build_pipeline_for_model(model_type, size, mode, backbone_type=backbone_type, model_name=model_name)

'''
def build_aug_transformation_pipeline(size: int, train: bool = True,
                                      model_type: str = "vit", backbone_type=None, model_name=None):
    mode = "train" if train else "eval"
    return build_aug_pipeline_for_model(model_type, size, mode, backbone_type=backbone_type, model_name=model_name)
'''

def build_aug_transformation_pipeline(
    size: int,
    train: bool = True,
    model_type: str = "vit",
    backbone_type=None,
    model_name=None,
    aug_id: int = 0,
):
    mode = "train" if train else "test"  # use "test" to match your style
    return build_aug_pipeline_for_model(
        model_type=model_type,
        size=size,
        mode=mode,
        backbone_type=backbone_type,
        model_name=model_name,
        aug_id=aug_id,
    )


def build_segmentation_transform(size: int, train: bool = True):
    return SegmentationTransform(size=size, train=train)
