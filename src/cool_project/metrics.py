from typing import Dict, Optional
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassAUROC,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
)
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
)


def build_metrics_for_task(
    task: str,
    num_classes: Optional[int] = None,
    top_k: int = 5,   # ✅ NEW
) -> Dict[str, nn.Module]:

    task = task.lower()
    metrics: Dict[str, nn.Module] = {}

    # ------------------------------------------------------
    # 1) Single-label classification
    # ------------------------------------------------------
    if task == "single_label_classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks.")

        if num_classes == 2:
            metrics["accuracy"] = BinaryAccuracy()
            metrics["auroc"] = BinaryAUROC()
        else:
            # Top-1 accuracy
            metrics["accuracy"] = MulticlassAccuracy(num_classes=num_classes)

            # ✅ Top-K Accuracy
            metrics[f"top{top_k}_accuracy"] = MulticlassAccuracy(
                num_classes=num_classes,
                top_k=top_k
            )

            metrics["auroc"] = MulticlassAUROC(num_classes=num_classes)

        return metrics

    # ------------------------------------------------------
    # 2) Multi-label classification
    # ------------------------------------------------------
    if task in ["multi_label", "multi_label_classification", "multilabel"]:
        if num_classes is None:
            raise ValueError("num_classes must be provided for multi-label tasks.")

        metrics["macro_accuracy"] = MultilabelAccuracy(
            num_labels=num_classes,
            average="macro",
        )
        metrics["micro_accuracy"] = MultilabelAccuracy(
            num_labels=num_classes,
            average="micro",
        )

        metrics["macro_f1"] = MultilabelF1Score(
            num_labels=num_classes,
            average="macro",
        )
        metrics["micro_f1"] = MultilabelF1Score(
            num_labels=num_classes,
            average="micro",
        )

        metrics["auroc"] = MultilabelAUROC(num_labels=num_classes)

        return metrics

    # ------------------------------------------------------
    # 3) Regression
    # ------------------------------------------------------
    if task == "regression":
        metrics["mse"] = MeanSquaredError()
        metrics["rmse"] = MeanSquaredError(squared=False)
        metrics["mae"] = MeanAbsoluteError()
        metrics["r2"] = R2Score()
        return metrics

    raise ValueError(f"Unknown task for metrics: {task}")
