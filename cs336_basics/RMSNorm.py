import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.g, mean=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """
        in_type = x.dtype
        x = x.to(torch.float32)

        res = torch.multiply(
            torch.div(x, torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)),
            self.g,
        )

        return res.to(in_type)
