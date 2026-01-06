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
from pathlib import Path
from typing import Dict


import logging

logger = logging.getLogger(__name__)


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
            raise RuntimeError(f"Error loading image {img_path}: {e}")

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
#DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))


class GenericCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_root: str,
        transform=None,
        return_one_hot: bool = False,
        validate_labels: bool = True,
        img_ext: str = ".jpg",
    ) -> None:

        self.csv_path = csv_path
        self.images_root = images_root
        self.transform = transform
        self.return_one_hot = return_one_hot
        self.img_ext = img_ext.lower()
        self._printed_getitem_debug = False


        # -------- Load CSV --------
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} entries from '{csv_path}'.")

        if "image" not in df.columns:
            raise ValueError("CSV must contain an 'image' column.")

        # -------- Label columns --------
        label_cols = [c for c in df.columns if c != "image"]
        if not label_cols:
            raise ValueError("No label columns found.")

        # -------- Convert labels to numeric --------
        for c in label_cols:
            df[c] = pd.to_numeric(df[c], errors="raise")
        df[label_cols] = df[label_cols].astype(float)

        # -------- Optional validation --------
        if validate_labels:
            for col in label_cols:
                if not set(df[col].unique()).issubset({0.0, 1.0}):
                    raise ValueError(f"Invalid values in column '{col}'.")

        self.label_cols = list(label_cols)

        # -------- Detect implicit "other" --------
        L = df[self.label_cols].to_numpy(dtype=np.float32)
        row_sums = L.sum(axis=1)

        self.has_implicit_other = (
            ((L == 0.0) | (L == 1.0)).all() and   # binary labels
            (row_sums <= 1).all() and             # single-label style
            (row_sums == 0).any()                 # some all-zero rows
        )

        if self.has_implicit_other:
            self.class_names = ["other"] + self.label_cols
            self.num_classes = 1 + len(self.label_cols)
        else:
            self.class_names = self.label_cols
            self.num_classes = len(self.label_cols)

        print(f"Detected {self.num_classes} classes: {self.class_names}")

        print("====== Dataset label interpretation ======")
        print(f"Implicit 'other' enabled: {self.has_implicit_other}")
        print(f"Label columns           : {self.label_cols}")
        print(f"Class names             : {self.class_names}")
        print(f"Number of classes       : {self.num_classes}")
        print("==========================================")


        # -------- Map image_id -> path --------
        id_to_path = {}
        for root, _, files in os.walk(images_root):
            for fname in files:
                if fname.lower().endswith(self.img_ext):
                    img_id = os.path.splitext(fname)[0]
                    id_to_path[img_id] = os.path.join(root, fname)

        if not id_to_path:
            raise RuntimeError("No images found.")

        # -------- Filter missing images --------
        mask = df["image"].isin(id_to_path.keys())
        df = df[mask].reset_index(drop=True)

        print(f"Final dataset size: {len(df)}")

        self.df = df
        self.id_to_path = id_to_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------- Image --------
        image_id = row["image"]
        img = Image.open(self.id_to_path[image_id]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # -------- Label --------
        label_array = row[self.label_cols].to_numpy(dtype=np.float32)

        # print once (only debug)
        if self.has_implicit_other and not self._printed_getitem_debug:
            print(f"[GenericCSVDataset] has_implicit_other={self.has_implicit_other}")
            print(f"[GenericCSVDataset] label_cols={self.label_cols}")
            self._printed_getitem_debug = True

        if self.has_implicit_other:
            s = int(label_array.sum())
            if s == 0:
                y = 0  # other
            elif s == 1:
                y = 1 + int(label_array.argmax())  # melanoma->1, sk->2
            else:
                raise ValueError("Multiple positives in implicit-other dataset")

            if self.return_one_hot:
                label = torch.zeros(self.num_classes, dtype=torch.float32)
                label[y] = 1.0
            else:
                label = int(y)

        else:
            # no implicit-other
            if self.return_one_hot:
                label = torch.from_numpy(label_array).float()
            else:
                label = int(label_array.argmax())

        return img, label, image_id



class DistillDataset(Dataset):
    """
    Wraps an existing dataset that returns (image, label)
    and replaces labels with precomputed teacher embeddings.
    """
    def __init__(self, base_dataset: Dataset, teacher_embs: torch.Tensor):
        self.base_dataset = base_dataset
        self.teacher_embs = teacher_embs

        if len(self.base_dataset) != len(self.teacher_embs):
            raise ValueError(
                f"Base dataset length {len(self.base_dataset)} != "
                f"teacher_embs length {len(self.teacher_embs)}"
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # base dataset returns (image_tensor, label)
        img, _ = self.base_dataset[idx]   # ignore original label
        teacher_emb = self.teacher_embs[idx]
        return img, teacher_emb


class DistillDatasetById(Dataset):
    """
    Safer distillation dataset.
    Matches teacher embeddings to samples by image_id.
    Intended for use with precomputed embeddings from a separate run.
    """

    def __init__(self, base_dataset, id_to_embedding: dict):
        self.base_dataset = base_dataset
        self.id_to_embedding = id_to_embedding

    def __len__(self):
        return len(self.base_dataset)

    def _get_id(self, idx):
        if hasattr(self.base_dataset, "get_id"):
            return str(self.base_dataset.get_id(idx))

        if hasattr(self.base_dataset, "image_ids"):
            return str(self.base_dataset.image_ids[idx])

        if hasattr(self.base_dataset, "df"):
            row = self.base_dataset.df.iloc[idx]
            for key in ["image", "image_id", "id", "filename"]:
                if key in row:
                    return str(row[key])

        raise RuntimeError(
            "Cannot infer image_id from base_dataset. "
            "Expose get_id(), image_ids, or df['image']."
        )

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img = item[0] if isinstance(item, (tuple, list)) else item

        image_id = self._get_id(idx)
        if image_id not in self.id_to_embedding:
            raise KeyError(f"No teacher embedding for image_id='{image_id}'")

        return img, self.id_to_embedding[image_id]

def load_teacher_id_map(embeddings_dir: Path, split: str) -> Dict[str, torch.Tensor]:
    """
    Loads <split>_embeddings.pt and returns a mapping image_id -> embedding tensor.
    """
    emb_path = embeddings_dir / f"{split}_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"[distill] Missing teacher embeddings file: {emb_path}")

    obj = torch.load(emb_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(
            f"[distill] Expected dict from {emb_path} (with keys embeddings/image_ids). Got: {type(obj)}"
        )

    embs = obj.get("embeddings", None)
    ids = obj.get("image_ids", None)

    if embs is None or ids is None:
        raise ValueError(
            f"[distill] {emb_path} must contain keys 'embeddings' and 'image_ids'. "
            f"Found keys: {list(obj.keys())}"
        )

    if not torch.is_tensor(embs) or embs.ndim != 2:
        raise ValueError(f"[distill] embeddings must be a tensor [N, D]. Got: {type(embs)} {getattr(embs, 'shape', None)}")

    if len(ids) != embs.shape[0]:
        raise ValueError(f"[distill] image_ids length {len(ids)} != embeddings rows {embs.shape[0]} for {emb_path}")

    # Store as CPU tensors; DataLoader will move x to GPU later in LightningModel if needed
    id_to_emb = {str(i): embs[k].float().cpu() for k, i in enumerate(ids)}
    logger.info(f"[distill] Loaded teacher map for split='{split}': N={len(id_to_emb)}, D={embs.shape[1]}")
    return id_to_emb

class SegmentationDataset(Dataset):
    """
    Dataset for segmentation: returns (image, mask).

    Expected layout:
        data/<dataset_name>/<split>/images/**/<img_files>
        data/<dataset_name>/<split>/masks/*.png
    """
    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        # collect images recursively
        image_paths = []
        for root, _, files in os.walk(image_dir):
            for fname in files:
                if fname.lower().endswith((".jpg")):
                    image_paths.append(os.path.join(root, fname))
        self.image_paths = sorted(image_paths)

        # collect masks (can be non-recursive if they’re all directly in masks/)
        mask_paths = []
        for root, _, files in os.walk(mask_dir):
            for fname in files:
                if fname.lower().endswith(".png"):
                    mask_paths.append(os.path.join(root, fname))
        self.mask_paths = sorted(mask_paths)

        self.transform = transform

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found under '{image_dir}' (recursively).")

        if len(self.mask_paths) == 0:
            raise ValueError(f"No masks found under '{mask_dir}' (recursively).")

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Number of images ({len(self.image_paths)}) "
                f"!= number of masks ({len(self.mask_paths)})"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = transforms.ToTensor()(img)
            mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


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

    print(f"[load_dataset] DATA_ROOT    = {DATA_ROOT}")
    print(f"[load_dataset] dataset_name = {dataset_name}")
    print(f"[load_dataset] split        = {split}")
    print(f"[load_dataset] csv_path     = {csv_path}")
    print(f"[load_dataset] images_root  = {images_root}")

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

def load_segmentation_dataset(dataset_name: str, split: str, transform=None) -> SegmentationDataset:
    dataset_dir = os.path.join(DATA_ROOT, dataset_name, split)
    image_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "masks")
    return SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)


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

def compute_class_weights(dataset): #ONLY FOR SINGLE LABEL DATASETS
    """
    Computes class_weights and sample_weights from a dataset that returns:
        (image, class_index)

    Returns:
        class_weights: Tensor [num_classes]
        sample_weights: Tensor [num_samples]
    """
    all_labels = []

    for i in range(len(dataset)):
        item = dataset[i]
        y = item[1]  # assume (image, label, ...)

        # If y is a Tensor one-hot vector, convert to class index
        if isinstance(y, torch.Tensor) and y.ndim > 0:
            y = int(y.argmax().item())
        else:
            y = int(y)

        all_labels.append(y)

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    meta_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    num_classes = getattr(meta_ds, "num_classes", int(labels_tensor.max().item() + 1))
    class_counts = torch.bincount(labels_tensor, minlength=num_classes)

    print(f"[compute_class_weights] num_classes={num_classes}")
    print(f"[compute_class_weights] class_counts={class_counts.tolist()}")

    # Avoid division by zero (if some class exists in CSV but no images)
    class_counts[class_counts == 0] = 1

    class_counts_f = class_counts.float()
    N = class_counts_f.sum()
    K = float(num_classes)

    class_weights = N / (K * class_counts_f)   # balanced weights
    class_weights = class_weights / class_weights.mean()  # optional but recommended

    sample_weights = class_weights[labels_tensor]

    print(f"[compute_class_weights] class_weights={class_weights.tolist()}")
    print(f"[compute_class_weights] sample_weights (first 10)={sample_weights[:10].tolist()}")


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
        print("[create_image_dataloader] balanced=True → using WeightedRandomSampler")
        sampler = make_balanced_sampler(dataset)
        shuffle = False  
    else:
        print("[create_image_dataloader] balanced=False → no sampler, shuffle =", shuffle)
        sampler = None
        
    persistent = num_workers > 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
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
    print(f"[create_image_data_loaders] Loaded train_ds with {len(train_ds)} samples.")

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
        print(f"[create_image_data_loaders] No 'val' folder found; splitting {val_split*100:.1f}% from train as validation.")
        train_ds, val_ds, _ = _split_dataset_random(
            train_ds,
            val_split=val_split,
            test_split=0.0,
        )
        print(f"[create_image_data_loaders] After split: train_ds={len(train_ds)} samples, val_ds={len(val_ds)} samples.")
        
        # ensure validation uses eval_transform
        #if eval_transform is not None:
            #val_ds.dataset.transform = eval_transform

    # --------- Load test (assumed to exist) ---------
    if test_transform is None:
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
            balanced=False, 
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

def create_segmentation_data_loaders(
    dataset_name: str,
    train_transform=None,
    eval_transform=None,
    test_transform=None,
    val_split: float = 0.0,
    batch_size: int = 4,
    num_workers: int = 4,
):
    loaders = {}

    # ---------- TRAIN ----------
    train_ds = load_segmentation_dataset(
        dataset_name=dataset_name,
        split="train",
        transform=train_transform,
    )
    print(f"[create_segmentation_data_loaders] Loaded train_ds with {len(train_ds)} samples.")

    # ---------- VAL (optional folder) ----------
    try:
        val_ds = load_segmentation_dataset(
            dataset_name=dataset_name,
            split="val",
            transform=eval_transform,
        )
        # If dataset exists but is empty, treat as missing
        if len(val_ds) == 0:
            raise ValueError("Empty val segmentation dataset.")
    except (FileNotFoundError, ValueError) as e:
        print(f"[create_segmentation_data_loaders] No usable 'val' folder ({e}); will use val_split if > 0.")
        val_ds = None

    # If no usable val folder but val_split>0, carve from train (IN MEMORY ONLY)
    if val_ds is None and val_split > 0.0:
        print(f"[create_segmentation_data_loaders] No 'val' folder; splitting {val_split*100:.1f}% from train.")
        train_ds, val_ds, _ = _split_dataset_random(
            train_ds,
            val_split=val_split,
            test_split=0.0,
        )
        print(f"[create_segmentation_data_loaders] After split: train={len(train_ds)}, val={len(val_ds)}")

        # ensure validation uses eval_transform if possible
        #if eval_transform is not None and hasattr(val_ds, "dataset"):
            #val_ds.dataset.transform = eval_transform

    # ---------- TEST ----------
    test_ds = load_segmentation_dataset(
        dataset_name=dataset_name,
        split="test",
        transform=test_transform,
    )

    # ---------- BUILD LOADERS ----------
    loaders["train"] = create_image_dataloader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        balanced=False,
        shuffle=True,
    )

    if val_ds is not None:
        loaders["val"] = create_image_dataloader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            balanced=False,
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

    if task in ["multi_label", "multi_label_classification", "multilabel"]:
        return True

    if loss_type in ["bce", "bcewithlogits", "bce_logits", "bce_multilabel"]:
        return True

    # Everything else: assume single-label (multi-class)
    return False


