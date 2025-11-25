import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class HighPassFilter(nn.Module):
    def __init__(self, cutoff: float = 0.5, inplace: bool = False):
        super().__init__()
        assert(cutoff >= 0.0 and cutoff < 1.0)
        self.cutoff = Tensor([cutoff]) 
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # filter out values below self.cutoff
        return F.relu(input * (input >= self.cutoff), inplace=self.inplace)
    

class LevelTrigger(nn.Module):
    def __init__(self, device: torch.device, threshold: float = 0.5):
        super().__init__()
        assert(threshold >= 0.0 and threshold < 1.0)
        self.threshold = Tensor([threshold]).to(device)

    def forward(self, input: Tensor) -> Tensor:
        # filter out values below self.cutoff
        return (input >= self.threshold).to(input.dtype)