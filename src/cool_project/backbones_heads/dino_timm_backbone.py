import torch
import torch.nn as nn
import timm


class TimmDinoBackbone(nn.Module):
    """
    Wraps a DINO ViT model from timm and returns embeddings for arbitrary image size
    (e.g. 512x512) as long as it's divisible by patch_size.
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        img_size: int = 512,
        pooling: str = "cls",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.pooling = pooling

        # Create the model with no classifier head (num_classes=0 → features only)
        try:
            self.model = timm.create_model(
                model_name,
                img_size=img_size,
                pretrained=True,
                num_classes=0,
            )
        except TypeError:
            # TinyViT doesn't accept img_size
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
            )

        self.embed_dim = self.model.num_features

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        feats = self.model.forward_features(x)

        if isinstance(feats, dict):
            if feats.get("gap", None) is not None:
                return feats["gap"]
            t = feats.get("x", None)
            if t is None:
                raise ValueError(f"Unknown feature dict keys: {feats.keys()}")
            feats = t

        # feats is now a tensor
        if feats.ndim == 3:          # [B, N, D]
            return feats[:, 0] if self.pooling == "cls" else feats.mean(dim=1)
        if feats.ndim == 4:          # [B, C, H, W]
            return feats.mean(dim=(2, 3))
        return feats                 # [B, D]

