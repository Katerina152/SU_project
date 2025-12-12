from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC
)
from typing import Optional, Dict
import torch.nn as nn

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC,
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelAUROC,
)

from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    ExplainedVariance,
)


from torchmetrics import JaccardIndex
from torchmetrics.segmentation import DiceScore

def build_metrics_for_task(
    task: str,
    num_classes: Optional[int] = None,
    top_k: Optional[int] = 5,
) -> Dict[str, nn.Module]:

    task = task.lower()
    metrics: Dict[str, nn.Module] = {}

    # ------------------------------------------------------
    # 1) Single-label classification (binary & multiclass)
    # ------------------------------------------------------
    if task == "single_label_classification":
        if num_classes is None:
            raise ValueError("num_classes is required for single_label_classification")

        # -------- Binary classification --------
        if num_classes == 2:
            metrics["accuracy"] = BinaryAccuracy()
            metrics["precision"] = BinaryPrecision()
            metrics["recall"] = BinaryRecall()
            metrics["f1"] = BinaryF1Score()
            metrics["auroc"] = BinaryAUROC()
            

        # -------- Multiclass classification --------
        else:
            metrics["accuracy"] = MulticlassAccuracy(num_classes=num_classes)

            # Top-K accuracy 
            if top_k is not None:
                metrics[f"top{top_k}_accuracy"] = MulticlassAccuracy(
                    num_classes=num_classes,
                    top_k=top_k,
                )

            metrics["precision_macro"] = MulticlassPrecision(
                num_classes=num_classes,
                average="macro",
            )
            metrics["precision_weighted"] = MulticlassPrecision(
                num_classes=num_classes,
                average="weighted",
            )

            metrics["recall_macro"] = MulticlassRecall(
                num_classes=num_classes,
                average="macro",
            )
            metrics["recall_weighted"] = MulticlassRecall(
                num_classes=num_classes,
                average="weighted",
            )

            metrics["f1_macro"] = MulticlassF1Score(
                num_classes=num_classes,
                average="macro",
            )
            metrics["f1_weighted"] = MulticlassF1Score(
                num_classes=num_classes,
                average="weighted",
            )

            metrics["auroc"] = MulticlassAUROC(num_classes=num_classes)
           
            

        return metrics

    # ------------------------------------------------------
    # 2) Multi-label classification
    # ------------------------------------------------------
    if task in ["multi_label", "multilabel", "multi_label_classification"]:
        if num_classes is None:
            raise ValueError("num_classes is required for multi-label tasks")

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
        metrics["explained_variance"] = ExplainedVariance()
        return metrics

    # ------------------------------------------------------
    # 4) Segmentation
    # ------------------------------------------------------
    if task == "segmentation":
        if num_classes is None:
            raise ValueError("num_classes is required for segmentation")

        # We will feed class indices [B, H, W] to both metrics.
        metrics["miou"] = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
        )

        metrics["dice"] = DiceScore(
            num_classes=num_classes,
            input_format="index",   
        )

        return metrics

    # ------------------------------------------------------
    # Unknown task
    # ------------------------------------------------------
    raise ValueError(f"Unknown task: {task}")
