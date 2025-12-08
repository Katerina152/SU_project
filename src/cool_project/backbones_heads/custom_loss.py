import torch
import torch.nn as nn
import torch.nn.functional as F 


class CustomLoss(nn.Module):
    def __init__(self, lamda_feat: float= 1.0, lamda_cos: float= 1.0):
        """
        Implements:
          L = lambda_feat * ||h_t - h_s||_2^2
              + lambda_cos * (1 - cos(h_t, h_s))
        where h_t and h_s are teacher and student embeddings.
        """
        super().__init__()
        
        self.lamda_feat = lamda_feat
        self.lamda_cos = lamda_cos

    def forward(self, student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
        """
        student_emb: [B, D] student embeddings (h_s)
        teacher_emb: [B, D] teacher embeddings (h_t)
        """
        if student_emb.shape != teacher_emb.shape:
            raise ValueError("Student and teacher embeddings must have the same shape"
                             f" but got {student_emb.shape} and {teacher_emb.shape}")
        
        # Feature matching (L2)
        l2_term = F.mse_loss(student_emb, teacher_emb, reduction="mean")

        # Angular / cosine alignment
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)   # [B]
        cos_term = (1.0 - cos_sim).mean()

        # custom operation/paper
        loss = self.lambda_feat * l2_term + self.lambda_cos * cos_term
        return loss

