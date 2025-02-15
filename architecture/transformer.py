import torch as t
import torch.nn as nn 
from architecture.block import TransformerBlock
from src.embed import Embed
from src.layer_norm import LayerNorm
from src.positional_embed import PosEmbed
from src.unembed import Unembed
from jaxtyping import Int, Float 
from torch import Tensor 
from utils.helper import Config
from utils.tests import rand_int_test, load_gpt2_test, reference_gpt2, tokens

class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits

if __name__ == "__main__":
    rand_int_test(DemoTransformer, [2, 4])
    load_gpt2_test(DemoTransformer, reference_gpt2, tokens)