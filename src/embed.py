import torch as t
import torch.nn as nn 
from utils.helper import Config
from utils.tests import rand_int_test, load_gpt2_test, reference_gpt2, tokens
from jaxtyping import Int, Float
from torch import Tensor

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]

if __name__ == "__main__":
    rand_int_test(Embed, [2, 4])
    load_gpt2_test(Embed, reference_gpt2.embed, tokens)