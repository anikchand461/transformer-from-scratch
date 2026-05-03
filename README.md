# transformer-from-scratch

A full Transformer architecture (encoder-decoder) built from first principles using PyTorch. Every component — from token embeddings to cross-attention — is implemented manually, without relying on any high-level transformer libraries.

> Understanding the core concepts of the Transformer model and implementing it from first principles.

---

## Architecture Overview

The implementation follows the original "Attention Is All You Need" paper and is structured as a clean set of modular layers:

```
Inputs → Encoder → (enc_out) ─────────────────┐
                                               ↓
Outputs (shifted right) → Decoder → Linear → Softmax → Output Probabilities
```

### Layers (`layers.py`)

| Class                      | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| `TokenEmbedding`           | Learned token embeddings, scaled by `√d_model`                  |
| `PositionalEncoding`       | Sinusoidal positional encodings added to embeddings             |
| `InputBlock`               | Combines `TokenEmbedding` + `PositionalEncoding`                |
| `SelfAttention`            | Single-head scaled dot-product attention                        |
| `MultiHeadAttention`       | Multi-head self-attention with final linear projection          |
| `MaskedMultiHeadAttention` | Multi-head attention with causal mask (decoder)                 |
| `CrossAttention`           | Cross-attention between decoder queries and encoder keys/values |
| `FeedForward`              | Position-wise FFN: `Linear → ReLU → Linear` (4× expansion)      |

### Encoder (`encoder.py`)

Stacks `num_layers` of:

```
InputBlock → [MultiHeadAttention → LayerNorm → FeedForward → LayerNorm] × N
```

### Decoder (`decoder.py`)

Stacks `num_layers` of:

```
InputBlock → [MaskedMHA → LayerNorm → CrossAttention → LayerNorm → FFN → LayerNorm] × N
```

### Transformer (`transformer.py`)

Combines the encoder and decoder with a final `Linear(d_model, vocab_size)` output projection.

---

## Project Structure

```
transformer-from-scratch/
├── layers.py          # All building-block modules
├── encoder.py         # Encoder stack
├── decoder.py         # Decoder stack
├── transformer.py     # Full Transformer model
├── utils.py           # Causal mask generation
├── run.py             # Training script
├── test.py            # Inference / testing script
└── pyproject.toml     # Project dependencies (uv)
```

---

## Setup

Requires Python `>=3.11`. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/anikchand461/transformer-from-scratch
cd transformer-from-scratch
uv sync
```

Dependencies: `torch`, `numpy>=2.4.4`

---

## Usage

**Train:**

```bash
uv run run.py
```

**Test / Inference:**

```bash
uv run test.py
```

---

## Key Implementation Notes

- Positional encoding uses the standard sinusoidal formulation: `sin` for even indices, `cos` for odd indices.
- The causal mask in `MaskedMultiHeadAttention` fills masked positions with `-1e9` before softmax so future tokens are effectively invisible.
- Cross-attention passes decoder states as queries, and encoder output as both keys and values.
- All sub-layer outputs use residual connections + `LayerNorm` (Post-LN style).
- `d_model` must be divisible by `num_heads`.

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
