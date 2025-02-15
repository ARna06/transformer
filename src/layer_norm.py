import torch as t
import torch.nn as nn 
from utils.helper import Config
from utils.tests import rand_float_test, load_gpt2_test, reference_gpt2, cache
from jaxtyping import Float
from torch import Tensor

class LayerNorm(nn.Module):
    def __init__(self, cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))
    
    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        expectation = residual.mean(dim=-1, keepdim=True)
        variance = (residual.var(dim=-1, keepdim=True, unbiased=False)+self.cfg.layer_norm_eps).sqrt()
        reduced = (residual - expectation)/variance
        return reduced * self.w + self.b

if __name__ == "__main__":
    rand_float_test(LayerNorm, [2, 4, 768])
    load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])