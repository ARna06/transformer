# Transformer from Scratch 

## Overview

This repository contains an implementation of a Transformer model built entirely using PyTorch's `torch.nn.Module`, without relying on higher-level libraries such as `torch.nn.Transformer`. The implementation follows the original **"Attention Is All You Need"** paper by Vaswani et al.

## Features

- [x] Implements self-attention and multi-head attention from scratch

- [x] Includes positional encoding

- [x] Fully connected feed-forward network

- [x] Layer normalization and dropout

- [x] Encoder-decoder architecture

- [x] Trained with a tiny dataset made by Neel Nanda, with the first 10K entries in the Pile (inspired by Stas' version for OpenWebText)

## Usage

### Dependencies

Project has a `pyproject.toml` with the dependencies listed, managed by poetry. Ensure `poetry` is installed otherwise:

```{bash}
pip install poetry
poetry install
``` 

### Training

For now the transformer only has the capability to train itself and provide autoregressive measures. Internal sampling for the transformer blocks is in WIP.

> It used `wandb` for training, when the main file is compiled, you need to provide your own API key and the model would start training itself, with the charts and graphs can be seen in the link provided while training.

## File Structure

```
├── architeture
|   ├── attention.py
|   ├── blocks.py
|   ├── mlp.py
|   ├── transformer.py
├── src
|   ├── embed.py
|   ├── layer_norm.py
|   ├── positional_embed.py
|   ├── unembed.py
├── utils
|   ├── helper.py
|   ├── tests.py
|   ├── train_helpers.py
```

## References

- "Attention Is All You Need" (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html