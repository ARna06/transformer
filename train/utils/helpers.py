from architecture.transformer import DemoTransformer
from utils.helper import Config
from dataclasses import dataclass
import datasets 
from utils.tests import reference_gpt2
from transformer_lens.utils import tokenize_and_concatenate
from torch.utils.data import DataLoader
from torch import Tensor
from jaxtyping import Int, Float

model_cfg = Config(
    debug=False,
    d_model=256,
    n_heads=4,
    d_head=64,
    d_mlp=1024,
    n_layers=2,
    n_ctx=256,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg)

@dataclass
class TransformerTrainingArgs:
    batch_size = 16
    epochs = 20
    max_steps_per_epoch = 200
    lr = 1e-3
    weight_decay = 1e-2
    wandb_project: str | None = "mimicformer"
    wandb_name: str | None = None


args = TransformerTrainingArgs()

dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
# print(dataset)
# print(dataset[0]["text"][:100])

tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=4,
)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(
    dataset_dict["train"], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    dataset_dict["test"], batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
)

first_batch = train_loader.dataset[: args.batch_size]

#print(first_batch.keys())
#print(first_batch["tokens"].shape)

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens