# Transformer Explanation (`transformer.py`)

This module implements a **Cross-Attention Transformer** designed to score and aggregate patch embeddings. Unlike a standard self-attention transformer, this architecture uses a set of learnable **Latent Queries** to attend to the input sequence (the patches).

## Key Components

### 1. `pos_enc_1d`
Generates standard sinusoidal positional encodings.
$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$
*Note: In the current IDPS usage, positional info might be handled differently or implicitly.*

### 2. `ScaledDotProductAttention`
The core attention mechanism.
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$

```python
    def compute_attn(self, q, k):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))
        return attn
```

### 3. `MultiHeadCrossAttention`
This is where the specificity lies. Instead of standard Self-Attention ($Q=K=V=X$), we have **Cross-Attention** where:
*   $K, V$ come from the Input $X$ (The patch embeddings).
*   $Q$ is a **Learnable Parameter** (`self.q`).

```python
        self.q = nn.Parameter(torch.empty((1, n_token, D)))
```

**Why Learnable Queries?**
The learnable queries act as "prototypes" or "detectors" that look for specific patterns in the input patches regardless of their position. For example, one query might learn to attend to "tumor-like" textures, effectively performing a soft clustering or detection.

**Dimensions:**
*   Input $X$: $(B, L, D)$ where $L$ is sequence length (number of patches).
*   Query $Q$: $(1, N_{token}, D)$ where $N_{token}$ is usually 1 or 2 (one aggregation token or one per task).
*   Output: $(B, N_{token}, D)$.

### 4. `Transformer` (The Wrapper)
Combines the Cross-Attention layer with a Feed-Forward Network (MLP).

#### Scoring (`get_scores`)
This is crucial for the **Scout** and **Pass 2 Score** step. It extracts the raw attention weights from the Cross-Attention layer.

```python
    def get_scores(self, x):
        attn = self.crs_attn.get_attn(x)
        # attn shape: (B, H, n_token, L)
        # Mean over heads (dim 1) and tokens (dim 2) -> (B, L)
        return attn.mean(dim=1).transpose(1, 2).mean(-1)
```
These scores represent "how much attention the model pays to each patch". High attention = High importance $\rightarrow$ Select this patch.

#### Forward (`forward`)
Aggregates the embeddings.
1.  Passes $X$ (patches) as Keys/Values and internal $Q$ as Queries to Cross-Attention.
2.  Result is a weighted sum of patch embeddings (weighted by attention).
3.  Passes through MLP.
4.  Output is a fixed-size vector representation of the whole slide (or selected patches).
