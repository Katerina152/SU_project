import torch
import torch.nn as nn
from transformers import ViTModel
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple
from .heads import init_head
import torch.nn.functional as F



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

    def _compute_loss(self, logits, labels, embeddings):
        if labels is None or embeddings is None:
            return None

        # custom loss
        if self.loss_type == "custom":
            if self.custom_loss_fn is None:
                raise ValueError("loss_type='custom' but no custom_loss_fn was provided.")
            return self.custom_loss_fn(embeddings, embeddings_teacher = None)

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
        loss = self._compute_loss(logits, labels, pooled)

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

class SegmentationViTWithHead(nn.Module):
    def __init__(self, config: Dict[str, Any], custom_loss_fn=None):
        super().__init__()

        backbone_cfg = config["backbone"]
        head_cfg = config["head"]

        # IMPORTANT: set task to "segmentation" from config
        self.task = config.get("task", "segmentation")
        self.loss_type = config.get("loss_type", "auto")
        self.custom_loss_fn = custom_loss_fn

        hf_model_name = backbone_cfg["hf_model_name"]
        self.num_layers_to_use = backbone_cfg.get("num_layers_to_use", None)

        # 1. backbone
        self.vit = ViTModel.from_pretrained(hf_model_name)

        if self.num_layers_to_use is not None:
            self.trim_backbone(self.num_layers_to_use)

        embed_dim = self.vit.config.hidden_size

        # for ViT, patch_size may be an int or (h, w)
        patch_size = self.vit.config.patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = tuple(patch_size)

        # 2. head: same Linear/MLP head, but now applied to patch tokens
        # head.output_dim must be num_segmentation_classes
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

    def _compute_loss(self, logits, labels, embeddings):
        if labels is None:
            return None

        # --- segmentation branch ---
        if self.task == "segmentation":
            # logits: [B, C, H_out, W_out], labels: [B, H_lab, W_lab] (long)
            loss_fn = nn.CrossEntropyLoss()

            # If the spatial size of logits doesn't match labels, upsample logits
            if logits.ndim == 4 and labels.ndim == 3:
                if logits.shape[2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=labels.shape[-2:],   # (H_lab, W_lab) e.g. (224, 224)
                        mode="bilinear",
                        align_corners=False,
                    )

            return loss_fn(logits, labels)


        # --- original classification/regression behaviour ---
        # custom loss
        if self.loss_type == "custom":
            if self.custom_loss_fn is None:
                raise ValueError("loss_type='custom' but no custom_loss_fn was provided.")
            return self.custom_loss_fn(embeddings, embeddings_teacher=None)

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
        pixel_values: torch.Tensor,      # [B, 3, H, W]
        labels: Optional[torch.Tensor] = None,  # [B, H, W] long
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

        last_hidden = outputs.last_hidden_state  # [B, 1 + N_patches, D]
        B, seq_len, D = last_hidden.shape

        # 1) drop CLS token → patch tokens only
        patch_tokens = last_hidden[:, 1:, :]     # [B, N_patches, D]
        N = patch_tokens.size(1)

        # 2) infer patch grid (assume square grid)
        H_p = W_p = int(N ** 0.5)
        if H_p * W_p != N:
            raise ValueError(f"Number of patches {N} is not a perfect square.")

        patch_tokens = patch_tokens.view(B, H_p, W_p, D)  # [B, H_p, W_p, D]

        # 3) apply head to each patch
        patch_logits = self.head(patch_tokens)           # [B, H_p, W_p, C]
        C = patch_logits.size(-1)

        # 4) permute to [B, C, H, W] for segmentation
        logits = patch_logits.permute(0, 3, 1, 2).contiguous()  # [B, C, H_p, W_p]

        # 5) you can define embeddings as mean over patches or something similar
        embeddings = patch_tokens.mean(dim=(1, 2))  # [B, D]

        loss = self._compute_loss(logits, labels, embeddings)

        if not return_dict:
            out = (logits, embeddings, outputs.hidden_states, outputs.attentions)
            return ((loss,) + out) if loss is not None else out

        return VisionOutput(
            loss=loss,
            logits=logits,          # [B, C, H, W]
            embeddings=embeddings,  # [B, D]
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
    
    if model_type == "seg_vit":
        return SegmentationViTWithHead(config=model_cfg)

    raise ValueError(f"Unknown model type: {model_type}")