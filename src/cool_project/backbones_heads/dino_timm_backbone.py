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

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)

        # dict output
        if isinstance(feats, dict):
            if "gap" in feats and feats["gap"] is not None:
                out = feats["gap"]                         # [B, D]
            elif "x" in feats and feats["x"] is not None:
                t = feats["x"]
                if t.ndim == 3:                            # [B, N, D]
                    out = t[:, 0] if self.pooling == "cls" else t.mean(dim=1)
                elif t.ndim == 4:                          # [B, C, H, W]
                    out = t.mean(dim=(2, 3))
                else:
                    raise ValueError(f"Unexpected feats['x'] shape: {t.shape}")
            else:
                raise ValueError(f"Unknown feature dict keys: {feats.keys()}")

        # If it's just a tensor, assume already pooled (B, D)
        return feats
