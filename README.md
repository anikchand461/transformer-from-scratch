<div align="center">

<h1>🤖 transformer-from-scratch</h1>

<p>A full <strong>Encoder-Decoder Transformer</strong> built from first principles using PyTorch.<br/>
Every component — from token embeddings to cross-attention — implemented manually, zero shortcuts.</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/uv-package%20manager-7C3AED?style=flat"/>
  <img src="https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-green?style=flat"/>
</p>

<p><em>Understanding the core concepts of the Transformer model and implementing it from first principles.</em></p>

</div>

---

## 📐 Architecture

```
                        ┌─────────────────────────────────┐
                        │            ENCODER               │
  Inputs                │  InputBlock (Embed + PosEnc)     │
    │                   │       ↓                          │
    └──────────────────▶│  MultiHeadAttention              │
                        │       ↓  (+ residual + norm)     │
                        │  FeedForward                     │
                        │       ↓  (+ residual + norm)     │
                        │   × N layers                     │
                        └──────────────┬──────────────────┘
                                       │ enc_out
                        ┌──────────────▼──────────────────┐
                        │            DECODER               │
  Outputs (shifted) ───▶│  InputBlock (Embed + PosEnc)     │
                        │       ↓                          │
                        │  MaskedMultiHeadAttention        │
                        │       ↓  (+ residual + norm)     │
                        │  CrossAttention ◀── enc_out      │
                        │       ↓  (+ residual + norm)     │
                        │  FeedForward                     │
                        │       ↓  (+ residual + norm)     │
                        │   × N layers                     │
                        └──────────────┬──────────────────┘
                                       ↓
                              Linear(d_model, vocab_size)
                                       ↓
                              Output Probabilities
```

---

## 🗂️ Project Structure

```
transformer-from-scratch/
│
├── transformer/                # Core package
│   ├── __init__.py
│   ├── layers.py               # All building-block modules
│   ├── encoder.py              # Encoder stack
│   ├── decoder.py              # Decoder stack
│   ├── transformer.py          # Full Transformer model
│   └── utils.py                # Causal mask generation
│
├── docs/                       # Handwritten derivation notes (PDF)
│   └── transformer-notes.pdf
│
├── run.py                      # Training script
├── test.py                     # Inference / testing script
├── pyproject.toml
└── README.md
```

---

## 🧱 Layers (`layers.py`)

<div align="center">

| Module                     | Role                                                       |
| :------------------------- | :--------------------------------------------------------- |
| `TokenEmbedding`           | Learned embeddings scaled by `√d_model`                    |
| `PositionalEncoding`       | Sinusoidal position encodings (sin/cos)                    |
| `InputBlock`               | `TokenEmbedding` + `PositionalEncoding` combined           |
| `SelfAttention`            | Single-head scaled dot-product attention                   |
| `MultiHeadAttention`       | Multi-head self-attention + linear projection              |
| `MaskedMultiHeadAttention` | Causal masked MHA (decoder self-attention)                 |
| `CrossAttention`           | Decoder queries attend to encoder keys/values              |
| `FeedForward`              | Position-wise FFN: `Linear → ReLU → Linear` (4× expansion) |

</div>

---

## ⚙️ Setup

**Requires Python `>=3.11` and [uv](https://github.com/astral-sh/uv)**

```bash
git clone https://github.com/anikchand461/transformer-from-scratch
cd transformer-from-scratch
uv sync
```

**Run training:**

```bash
uv run run.py
```

**Run inference/test:**

```bash
uv run test.py
```

---

## 📝 Implementation Notes

- **Positional Encoding** — standard sinusoidal: `sin` for even indices, `cos` for odd indices across `d_model` dimensions
- **Causal Mask** — masked positions filled with `-1e9` before softmax, effectively zeroing future token attention
- **Cross-Attention** — decoder hidden states → Q, encoder output → K and V
- **Residual + LayerNorm** — applied after every sub-layer (Post-LN, matching the original paper)
- **Constraint** — `d_model` must be divisible by `num_heads`

---

## 📚 Documentation

Detailed handwritten derivations, mathematical walkthrough, and step-by-step code explanations are in [`docs/transformer-notes.pdf`](./docs/transformer-notes.pdf).

---

## 📖 References

<div align="center">

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 &nbsp;|&nbsp; [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar

</div>
