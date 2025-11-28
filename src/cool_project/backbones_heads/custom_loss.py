import torch.nn as nn

class CustomLoss(nn.Module):
    def forward(self, x):
        # custom operation/paper
        return x
