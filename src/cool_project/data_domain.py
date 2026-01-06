from typing import Dict
from .dataloader import create_image_data_loaders, load_segmentation_dataset, create_segmentation_data_loaders
from .preprocessing import build_transformation_pipeline, build_segmentation_transform
import logging

logger = logging.getLogger(__name__)

DOMAIN_DATASET_MAP = {
    "dermatology": "ISIC2019",
    "pathology": "TCGA",
    "radiology": "CheXpert",
    "derm_segmentation": "ISIC2019",
    "pathology_segmentation": "TCGA",
    "radiology_segmentation": "CheXpert",
}

def create_domain_loaders(
    domain: str,
    dataset_name: str | None,  
    resolution: int,
    batch_size: int,
    num_workers: int,
    balanced_train: bool = False,
    val_split: float = 0.0,
    return_one_hot: bool = False,
    model_type: str = "vit",
    backbone_type: str | None = None,
    model_name: str | None = None,

):
    if domain not in DOMAIN_DATASET_MAP:
        raise ValueError(f"Unknown domain: {domain}")

    default_name = DOMAIN_DATASET_MAP[domain]
    dataset_name = dataset_name or default_name

    if domain.endswith("_segmentation"):
        train_transform = build_segmentation_transform(resolution, train=True)
        eval_transform  = build_segmentation_transform(resolution, train=True)
        test_transform  = build_segmentation_transform(resolution, train=False)

        logger.info(f"[{domain}] (SEG) Train transform:\n{train_transform}")
        logger.info(f"[{domain}] (SEG) Eval  transform:\n{eval_transform}")
        logger.info(f"[{domain}] (SEG) Test  transform:\n{test_transform}")

        loaders = create_segmentation_data_loaders(
            dataset_name=dataset_name,
            train_transform=train_transform,
            eval_transform=eval_transform,
            test_transform=test_transform,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return loaders


    train_transform = build_transformation_pipeline(resolution, train=True,model_type=model_type, backbone_type=backbone_type, model_name=model_name)
    eval_transform  = build_transformation_pipeline(resolution, train=True, model_type=model_type, backbone_type=backbone_type, model_name=model_name)
    test_transform = build_transformation_pipeline(resolution, train=False, model_type=model_type, backbone_type=backbone_type, model_name=model_name)

    logger.info(f"[{domain}] Train transform:\n{train_transform}")
    logger.info(f"[{domain}] Eval  transform:\n{eval_transform}")
    logger.info(f"[{domain}] Test  transform:\n{test_transform}")


    loaders = create_image_data_loaders(
        dataset_name=dataset_name,
        train_transform=train_transform,
        eval_transform=eval_transform,
        test_transform=test_transform,
        val_split=val_split, 
        batch_size=batch_size,
        num_workers=num_workers,
        balanced_train=balanced_train,
        return_one_hot=return_one_hot, 
       
    )

    return loaders

