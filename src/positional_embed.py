import torch as t
import torch.nn as nn 
import einops
from utils.helper import Config
from utils.tests import rand_int_test, load_gpt2_test, reference_gpt2, tokens
from jaxtyping import Int, Float
from torch import Tensor

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)

if __name__ == "__main__":
    rand_int_test(PosEmbed, [2, 4])
    load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)