import math
import torch.nn as nn
import torch
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None): 
        """    
        Construct a linear transformation module. 
        This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply the linear transformation to the input.
        """
        return rearrange(
            torch.matmul(
                rearrange(x, '... d_in -> (...) d_in'),
                self.weight.t()
            ),
            '(b) d_out -> b d_out',
            b=x.shape[:-1].numel()
        ).reshape(*x.shape[:-1], self.out_features)