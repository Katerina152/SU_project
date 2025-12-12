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
        self.model = timm.create_model(
            model_name,
            img_size=img_size,    # this lets it know about 512
            pretrained=True,
            num_classes=0,        # no classification head, just features
        )
        self.embed_dim = self.model.num_features

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W), e.g. (B, 3, 512, 512)
        returns: (B, D) embeddings
        """
        feats = self.model.forward_features(x)

        if isinstance(feats, dict):
            # timm ViTs often return dicts
            if self.pooling == "cls":
                x_tokens = feats.get("x", None)
                if x_tokens is not None:
                    return x_tokens[:, 0]  # CLS token
            if "gap" in feats:
                return feats["gap"]
            if "x" in feats:
                return feats["x"].mean(dim=1)
            raise ValueError(f"Unknown feature dict keys: {feats.keys()}")

        # If it's just a tensor, assume already pooled (B, D)
        return feats
