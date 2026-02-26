# NanoGPT with Hyper-Connections

A modified implementation of [NanoGPT](https://github.com/karpathy/nanoGPT) featuring [Hyper-Connections](https://arxiv.org/abs/2409.19606), with several additional modern architecture improvements.

## Overview

This project integrates hyper-connections into NanoGPT's transformer architecture. Hyper-connections replace the standard residual stream with multiple parallel residual streams, increasing the model's information-carrying capacity without a proportional increase in depth or compute. Rather than a single stream of representations flowing through the network, the model maintains 4 residual streams simultaneously — allowing different layers to attend to and write to different parts of the representational space.

The hyper-connection mechanism learns dynamic, weighted mixing of these streams at both the input (width connection) and output (depth connection) of each branch, using lightweight learned parameters on top of a sensible static initialization.

## Architecture Changes

Compared to the baseline NanoGPT, this implementation includes the following modifications:

- **Hyper-Connections** (4 residual streams, 4 fractions) via [`hyper-connections`](https://github.com/lucidrains/hyper-connections)
- **RMSNorm** in place of LayerNorm
- **RoPE** (Rotary Position Embeddings) in place of learned position embeddings
- **SwiGLU** activation in the MLP in place of GeLU
- Biases disabled for all linear layers and norms
- Model dimension reduced to 512, attention heads reduced to 8

## Results

Both models were trained on the same dataset and evaluated using bits-per-character (BPC).

| Model | Params | BPC |
|---|---|---|
| This model (24L, d=512, hyper-connections) | ~100M | **1.14** |
| Baseline NanoGPT (same depth/context/hyperparams) | ~150M | 1.20 |

The hyper-connections model achieves better BPC with ~33% fewer parameters. The baseline's parameter count is higher primarily because it requires a wider model to match representational capacity — the extra streams in this model do that work more efficiently.

## Implementation

The core components are split across four files:

- `hyperconnections.py` — Hyper-connections and frac-connections logic (from [lucidrains/hyper-connections](https://github.com/lucidrains/hyper-connections))
- `hyperblock.py` — Transformer block wrapping attention and MLP with hyper-connections
- `modules.py` — CausalSelfAttention (with RoPE + Flash Attention), SwiGLU, MLP, RMSNorm
- `model.py` — GPT model: stream expansion, transformer stack, stream reduction, LM head

The model fans out to multiple streams after the token embedding, passes through `HyperBlock` layers, then reduces the streams back to one before the final norm and LM head.

## Usage
```python
from model import GPT, GPTConfig

config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=24,
    n_head=8,
    n_embd=512,
    dropout=0.1,
    bias=False,
    num_streams=4,
    num_fracs=4,
    use_swiglu=True,
)

model = GPT(config)
```

## References

- [Hyper-Connections (Zhu et al., 2024)](https://arxiv.org/abs/2409.19606)
- [Frac-Connections (2025)](https://arxiv.org/abs/2503.14125)
- [NanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)
- [lucidrains/hyper-connections](https://github.com/lucidrains/hyper-connections)
