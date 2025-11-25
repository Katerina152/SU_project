import torch
import torch.nn as nn
from transformers import ViTModel
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple
from .heads import init_head


@dataclass
class VisionOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    embeddings: torch.Tensor  
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class VisionTransformerWithHead(nn.Module):
    def __init__(self, config: Dict[str, Any], custom_loss_fn=None):
        super().__init__()

        backbone_cfg = config["backbone"]
        head_cfg = config["head"]

        self.task = config.get("task", "single_label_classification")
        self.loss_type = config.get("loss_type", "auto")
        self.custom_loss_fn = custom_loss_fn  # callable or None

        hf_model_name = backbone_cfg["hf_model_name"]
        self.pooling = backbone_cfg.get("pooling", "cls")
        self.num_layers_to_use = backbone_cfg.get("num_layers_to_use", None)

        # 1. backbone
        self.vit = ViTModel.from_pretrained(hf_model_name)

        if self.num_layers_to_use is not None:
            self.trim_backbone(self.num_layers_to_use)

        embed_dim = self.vit.config.hidden_size

        # 2. head
        self.head = init_head(head_cfg, input_dim=embed_dim)

        # 3. freeze if requested
        if backbone_cfg.get("freeze_backbone", False):
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = True

    def trim_backbone(self, num_layers: int):
        encoder_layer = self.vit.encoder.layer
        self.vit.encoder.layer = encoder_layer[:num_layers]

    def _compute_loss(self, logits, labels):
        if labels is None:
            return None

        # custom loss
        if self.loss_type == "custom":
            if self.custom_loss_fn is None:
                raise ValueError("loss_type='custom' but no custom_loss_fn was provided.")
            return self.custom_loss_fn(logits, labels)

        # map 'auto' to standard based on task
        loss_type = self.loss_type
        if loss_type == "auto":
            if self.task == "regression":
                loss_type = "mse"
            elif self.task == "single_label_classification":
                loss_type = "ce"
            elif self.task == "multi_label_classification":
                loss_type = "bce"
            else:
                raise ValueError(f"Unknown task: {self.task}")

        if loss_type == "mse":
            loss_fn = nn.MSELoss()
            if logits.ndim == 1:
                return loss_fn(logits.squeeze(), labels.squeeze())
            else:
                return loss_fn(logits, labels)

        if loss_type == "ce":
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        if loss_type == "bce":
            loss_fn = nn.BCEWithLogitsLoss()
            return loss_fn(logits, labels)

        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[VisionOutput, Tuple]:

        outputs = self.vit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden]
        if self.pooling == "cls":
            pooled = last_hidden[:, 0]
        elif self.pooling == "mean":
            pooled = last_hidden.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.head(pooled)
        loss = self._compute_loss(logits, labels)

        if not return_dict:
            out = (logits, pooled, outputs.hidden_states, outputs.attentions)
            return ((loss,) + out) if loss is not None else out

        return VisionOutput(
            loss=loss,
            logits=logits,
            embeddings=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class DinoWithHead(nn.Module):
    def __init__(self, config: Dict[str, Any], custom_loss_fn=None):
        super().__init__()

        
def build_model_from_config(cfg):
    model_cfg = cfg["model"]
    model_type = model_cfg.get("type", "vit").lower()

    if model_type == "vit":
        return VisionTransformerWithHead(config=model_cfg)

    if model_type == "dino":
        return DinoWithHead(config=model_cfg)

    raise ValueError(f"Unknown model type: {model_type}")