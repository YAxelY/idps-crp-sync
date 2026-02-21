# Full Sequential Architecture Flow

This document describes the complete lifecycle of a single training step in the IDPS-CRP framework, tracing a Gigapixel Image from raw sparse data to the final Loss computation.

---

## 1. Data Loading Phase (CPU)
*Script: `data/megapixel_mnist/mnist_dataset.py`*

1.  **Sparse Retrieval**: The Dataset loader reads the list of non-zero indices and values for the image $I$.
2.  **Dense Reconstruction**: An empty $H \times W$ tensor is allocated and filled with values at the specified indices.
3.  **Patching (Unfold)**: The image is sliced into a grid of patches $P = \{p_1, p_2, ..., p_N\}$.
    *   Example: $1500 \times 1500$ image $\rightarrow$ $30 \times 30 = 900$ patches of size $50 \times 50$.
4.  **Batch Generation**: A batch of samples is created.
    *   **Scout Input**: A low-resolution view (downsampled) or the full set (if manageable).
    *   **Learner Input**: Access to high-resolution patches on demand.
    *   *Note: In this implementation, `get_scout_data` returns the low-res version of all patches.*

---

## 2. Pass 1: The Scout (GPU - Contextual Filtering)
*Script: `architecture/idps_net.py` -> `scouting_pass`*

**Goal**: Reduce the search space from $N$ (e.g., 900) to $M$ (e.g., 50) candidates.

1.  **Input**: Batch of Low-Res patches $(B, N, C, h, w)$.
2.  **Backbone (ResNet)**: Each patch is encoded into a vector $v_i$.
3.  **Transformer Scoring**:
    *   The embeddings $V = \{v_1, ..., v_N\}$ are fed to the Transformer.
    *   Cross-Attention with a learnable Query $Q$ computes attention weights $A \in \mathbb{R}^{B \times N}$.
    *   $A[i]$ represents the "importance" of patch $p_i$.
4.  **Top-M Selection (Hard)**:
    *   We pick the indices of the top $M$ scores: $\text{idx}_{top} = \text{topk}(A, M)$.
    *   This step is **Non-Differentiable**, but acceptable because we only use it to prune the search space.

---

## 3. Pass 2: The Learner (GPU - Differentiable Selection)
*Script: `architecture/idps_net.py` -> `training_pass`*

**Goal**: Select best $K$ from $M$ and classify.

1.  **HD Extraction**: Using $\text{idx}_{top}$ from Pass 1, we fetch the corresponding High-Resolution patches from the dataset.
    *   Input: Candidates $(B, M, C, H, W)$ (50 patches).
2.  **Backbone (ResNet)**: Re-encode these 50 patches (Gradient enabled for fine-tuning).
    *   Embeddings $E_{cand} \in \mathbb{R}^{B \times M \times D}$.
3.  **Transformer Scoring (Again)**:
    *   Compute importance scores $S \in \mathbb{R}^{B \times M}$.
    *   **Normalization**: Scores are normalized to $[0, 1]$ to stabilize noise addition.
4.  **DPS Selection (Module `dps_module.py`)**:
    *   **Add Noise**: $\tilde{S} = S + \sigma \cdot \epsilon$.
    *   **Top-K**: Find top $K$ (e.g., 10) in the perturbed scores.
    *   **Indicator**: Create soft one-hot matrix $I_{dps} \in \mathbb{R}^{B \times K \times M}$.
    *   *Crucial*: The backward pass of this module allows gradients to flow from the selection back to the scoring mechanism.
5.  **Soft Gather**:
    *   $E_{selected} = I_{dps} \times E_{cand}$. Result shape $(B, K, D)$.
6.  **Aggregation**:
    *   The selected embeddings are aggregated by the Transformer into a slide-level representation $Z \in \mathbb{R}^{B, D}$.
7.  **Classification**:
    *   Head: $Y_{pred} = \text{Linear}(Z)$.

---

## 4. Loss & Backpropagation
*Script: `training/iterative.py`*

1.  **Compute Loss**: $L = \text{CrossEntropy}(Y_{pred}, Y_{gt})$.
2.  **Backward**: $\nabla L$ propagates backwards:
    *   Through the Classifier Head...
    *   Through the Aggregator...
    *   Through the **DPS Selector** (updating the Scoring Transformer to pick better patches)...
    *   Through the ResNet Encoder (improving feature extraction).

## Summary
The pipeline transforms a massive input problem into a manageable two-step process: **Coarse Filtering (Scout)** followed by **Differentiable Fine Selection (Learner)**, ensuring that the model learns *where to look* (Scoring) and *what to see* (Encoding) simultaneously.
