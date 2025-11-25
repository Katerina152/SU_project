from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import os
import torch
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import pandas as pd
from typing import List
from torch.utils.data import random_split
import torch.nn as nn  

# Load Dataset for raw images, self-supervised learning/image ..... 
class Image_Dataset(Dataset):
    def __init__(self, image_folder: str, transform=None):
        """
        Dataset that just loads all .jpg images from a folder.
        No labels or CSV.
        """
        self.paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        if self.transform is not None:
            img = self.transform(img)

        return img


# This file is expected to live in: <repo>/src/repo/dataloader.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# One level up: <repo>/src
SRC_ROOT = os.path.dirname(THIS_DIR)

# Two levels up: <repo>
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

# Data directory: <repo>/data
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")


class GenericCSVDataset(Dataset):
    """
    Generic dataset for:
      - a CSV with one 'image' column + numeric label columns (one-hot)
      - image files stored under a root folder (possibly in subfolders)

    Expected layout for a dataset called <dataset_name>:

        <repo>/data/<dataset_name>/<split>/labels.csv
        <repo>/data/<dataset_name>/<split>/images/.../*.jpg

    Example for ISIC2019:

        repo/
          src/
          data/
            ISIC2019/
              train/
                labels.csv
                images/
                  ISIC_2019_Training_Input/
                    ISIC_0000000.jpg
                    ...
              test/
                labels.csv
                images/
                  ...
    """

    def __init__(
        self,
        csv_path: str,
        images_root: str,
        transform=None,
        return_one_hot: bool = False,
        validate_labels: bool = True,
        img_ext: str = ".jpg",
    ) -> None:
        """
        Args:
            csv_path: Path to CSV file with 'image' column and label columns.
            images_root: Root directory under which images are stored.
                         Images may be inside nested subfolders.
            transform: Optional torchvision-style transform for images.
            return_one_hot:
                - False -> returns (image, class_index) for CrossEntropyLoss.
                - True  -> returns (image, one_hot_vector) e.g. for BCEWithLogitsLoss.
            validate_labels: If True, ensure labels are only 0/1 after type conversion.
            img_ext: Image extension to look for (default ".jpg").
        """
        self.csv_path = csv_path
        self.images_root = images_root
        self.transform = transform
        self.return_one_hot = return_one_hot
        self.img_ext = img_ext.lower()

        # -------- Load CSV --------
        df = pd.read_csv(csv_path)

        
        # -------- Basic sanity: must have an 'image' column --------
        if "image" not in df.columns:
            raise ValueError(f"CSV '{csv_path}' must contain an 'image' column.")

        # -------- Label columns = all columns except 'image' --------
        label_cols = [c for c in df.columns if c != "image"]
        if not label_cols:
            raise ValueError(
                f"No label columns found in '{csv_path}'. "
                "Expected one 'image' column and at least one label column."
            )

        # -------- Convert label columns to numeric --------
        for c in label_cols:
            df[c] = pd.to_numeric(df[c], errors="raise")  # error if something is non-numeric

        df[label_cols] = df[label_cols].astype(float)

        # -------- Optional: validate that labels are only 0/1 --------
        if validate_labels:
            for col in label_cols:
                unique_vals = set(df[col].unique())
                if not unique_vals.issubset({0.0, 1.0}):
                    raise ValueError(
                        f"Column '{col}' in '{csv_path}' contains invalid values: {unique_vals}. "
                        "Expected only 0 or 1 after conversion to float."
                    )

        # -------- Save metadata on the dataset --------
        self.label_cols = label_cols            
        self.class_names = label_cols
        self.num_classes = len(label_cols)        

        # -------- Map image_id -> full path, scanning all subfolders --------
        id_to_path = {}
        for root, _, files in os.walk(images_root):
            for fname in files:
                if fname.lower().endswith(self.img_ext):
                    img_id = os.path.splitext(fname)[0]  # 'ISIC_0000000'
                    full_path = os.path.join(root, fname)
                    id_to_path[img_id] = full_path

        if len(id_to_path) == 0:
            raise RuntimeError(
                "No images with extension '{}' found under '{}'."
                .format(self.img_ext, images_root)
            )

        # -------- Keep only rows whose image exists on disk --------
        mask = df["image"].isin(id_to_path.keys())
        if not mask.all():
            missing = df.loc[~mask, "image"].tolist()
            print(
                "Warning: {} images listed in '{}' were not found under '{}'. "
                "They will be skipped.".format(len(missing), csv_path, images_root)
            )

        df = df[mask].reset_index(drop=True)

        self.df = df
        self.id_to_path = id_to_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # -------- Image --------
        image_id = row["image"]
        image_path = self.id_to_path[image_id]

        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # -------- Label --------
        # Ensure numeric labels as float32 (handles any lingering object dtype)
        label_array = row[self.label_cols].to_numpy(dtype=np.float32)
        label_vector = torch.from_numpy(label_array)


        if self.return_one_hot:
            # multi-class one-hot / multi-label
            label = label_vector
        else:
            # single-label: index of max value
            label = int(label_vector.argmax().item())

        return img, label

# ------------------------------------------------------------
# High-level helper: load by dataset name + split only
# ------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    split: str = "train",
    transform=None,
    return_one_hot: bool = False,
    img_ext: str = ".jpg",
) -> GenericCSVDataset:
    """
    Load a dataset located under the repo 'data/' folder using only its name.

    Expected layout for a dataset called <dataset_name>:

        <repo>/data/<dataset_name>/<split>/labels.csv
        <repo>/data/<dataset_name>/<split>/images/   (may contain subfolders)

    Example (ISIC2019):

        repo/
          src/
          data/
            ISIC2019/
              train/
                labels.csv
                images/
                  ISIC_2019_Training_Input/
                    ISIC_0000000.jpg
                    ...
              test/
                labels.csv
                images/
                  ...

    Args:
        dataset_name: Name of the dataset folder inside 'data/' (e.g. 'ISIC2019').
        split: 'train', 'val', 'test', etc. (must match subfolder name).
        transform: Optional torchvision transform for images.
        return_one_hot:
            - False -> labels as class index (int).
            - True  -> labels as one-hot/multi-hot float vector.
        img_ext: Image file extension (default '.jpg').

    Returns:
        A GenericCSVDataset instance.
    """
    dataset_dir = os.path.join(DATA_ROOT, dataset_name, split)
    csv_path = os.path.join(dataset_dir, "labels.csv")
    images_root = os.path.join(dataset_dir, "images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "CSV file not found at '{}'. "
            "Expected layout: data/{}/{}/labels.csv"
            .format(csv_path, dataset_name, split)
        )
    if not os.path.isdir(images_root):
        raise FileNotFoundError(
            "Images root folder not found at '{}'. "
            "Expected layout: data/{}/{}/images/"
            .format(images_root, dataset_name, split)
        )

    return GenericCSVDataset(
        csv_path=csv_path,
        images_root=images_root,
        transform=transform,
        return_one_hot=return_one_hot,
        img_ext=img_ext,
    )


# Load Dataset for precomputed embeddings
class Embedding_Dataset(Dataset):
    def __init__(self, embedding_folder):
        self.paths = sorted(glob.glob(os.path.join(embedding_folder, "*.pt")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        emb_path = self.paths[idx]
        #emb = np.load(emb_path) 
        emb = torch.load(emb_path)          
        emb = torch.tensor(emb, dtype=torch.float32)
        return emb




#DataLoaders and weighted samplers for class imbalance

def compute_class_weights(dataset):
    """
    Computes class_weights and sample_weights from a dataset that returns:
        (image, class_index)

    Returns:
        class_weights: Tensor [num_classes]
        sample_weights: Tensor [num_samples]
    """
    all_labels = []

    for i in range(len(dataset)):
        _, y = dataset[i]

        # If y is a Tensor one-hot vector, convert to class index
        if isinstance(y, torch.Tensor) and y.ndim > 0:
            y = int(y.argmax().item())
        else:
            y = int(y)

        all_labels.append(y)

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    num_classes = int(labels_tensor.max().item() + 1)
    class_counts = torch.bincount(labels_tensor, minlength=num_classes)

    # Avoid division by zero (if some class exists in CSV but no images)
    class_counts[class_counts == 0] = 1

    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels_tensor]

    return class_weights, sample_weights

def make_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler that oversamples minority classes.
    """
    _, sample_weights = compute_class_weights(dataset)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def create_image_dataloader(dataset, batch_size=32, num_workers=4, balanced=False, shuffle=True):
    """
    Creates a DataLoader with optional class-balancing via WeightedRandomSampler.
    """
    if balanced:
        sampler = make_balanced_sampler(dataset)
        shuffle = False  
    else:
        sampler = None
        

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

def _split_dataset_random(dataset, val_split: float = 0.2, test_split: float = 0.0):
    """
    Internal helper:
      - Takes a Dataset
      - Randomly splits into train / val / test according to fractions.

    val_split + test_split must be < 1.0.
    If val_split == 0.0, returns (train, None, test or None).
    If test_split == 0.0, returns (train, val or None, None).
    """
    if val_split < 0 or test_split < 0:
        raise ValueError("val_split and test_split must be >= 0.")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0.")

    n_total = len(dataset)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = None, None, None

    if n_test > 0:
        train_ds, test_ds = random_split(dataset, [n_total - n_test, n_test])
    else:
        train_ds = dataset

    if n_val > 0:
        n_train_final = len(train_ds) - n_val
        train_ds, val_ds = random_split(train_ds, [n_train_final, n_val])

    return train_ds, val_ds, test_ds


def create_image_data_loaders(
    dataset_name: str,
    train_transform=None,
    eval_transform=None,
    test_transform=None,
    val_split=0.0, 
    batch_size: int = 32,
    num_workers: int = 4,
    balanced_train: bool = False,   
    img_ext: str = ".jpg",
    return_one_hot: bool = False,
):
    """
    High-level helper that wraps GenericCSVDataset + create_image_dataloader.

    Assumptions:
      - data/<dataset_name>/train/ exists
      - data/<dataset_name>/test/ exists
      - data/<dataset_name>/val/ is OPTIONAL

    Behavior:
      - If val/ folder exists:
          use it as validation set.
      - If val/ folder does NOT exist and val_split > 0:
          take a random fraction of the train set as validation.
      - Test set is always taken from 'test' folder and is NOT split.

    Returns:
        dict with keys: "train", optionally "val", and "test".
    """
    loaders = {}

    # --------- Load train ---------
    train_ds = load_dataset(
        dataset_name=dataset_name,
        split="train",
        transform=train_transform,
        return_one_hot=return_one_hot,
        img_ext=img_ext,
    )

    # --------- Load optional val ---------
    try:
        val_ds = load_dataset(
            dataset_name=dataset_name,
            split="val",
            transform=eval_transform,
            return_one_hot=return_one_hot,
            img_ext=img_ext,
        )
    except FileNotFoundError:
        val_ds = None

    # --------- If no val folder but val_split > 0, carve val from train ---------
    if val_ds is None and val_split > 0.0:
        # use your helper here; we keep test_split=0.0 to NOT touch disk test
        train_ds, val_ds, _ = _split_dataset_random(
            train_ds,
            val_split=val_split,
            test_split=0.0,
        )

        # ensure validation uses eval_transform
        if eval_transform is not None:
            # train_ds and val_ds share the same underlying dataset,
            # so updating val_ds.dataset.transform is enough
            val_ds.dataset.transform = eval_transform

    # --------- Load test (assumed to exist) ---------
    if test_transform is None:
        # often you want test to behave like val by default
        test_transform = eval_transform


    # --------- Load test (assumed to exist) ---------
    test_ds = load_dataset(
        dataset_name=dataset_name,
        split="test",
        transform=test_transform,
        return_one_hot=return_one_hot,
        img_ext=img_ext,
    )


    # --------- Build DataLoaders ---------
    loaders["train"] = create_image_dataloader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        balanced=balanced_train,
        shuffle=True,
    )

    if val_ds is not None:
        loaders["val"] = create_image_dataloader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            balanced=False,  # usually no balancing for val
            shuffle=False, 
        )

    loaders["test"] = create_image_dataloader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        balanced=False,
        shuffle=False,  
    )

    return loaders

def is_multilabel_from_config(cfg: dict) -> bool:
    """
    Decide whether this is a multi-label problem based on the config.
    """
    task = cfg.get("task", "").lower()
    loss_type = cfg.get("model", {}).get("loss_type", "").lower()

    # If the task explicitly says multi-label, trust it.
    if task in ["multi_label", "multi_label_classification", "multilabel"]:
        return True

    # If the loss is a typical multi-label loss, also treat it as multi-label.
    if loss_type in ["bce", "bcewithlogits", "bce_logits", "bce_multilabel"]:
        return True

    # Everything else: assume single-label (multi-class)
    return False

