import torch.nn as nn

class CustomActivation(nn.Module):
    def forward(self, x):
        # your custom operation
        return x
