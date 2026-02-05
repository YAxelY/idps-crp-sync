# IDPS-CRP Method Description

## 1. Overview

**IDPS-CRP** (Iterative Patch Selection with Sparse Re-computation for Concept Relevance Propagation) is a specialized Deep Learning architecture for classifying Gigapixel Whole Slide Images (WSI) in Computational Pathology.

It addresses the fundamental trade-off between **Computational Efficiency** (memory constraints) and **Model Interpretability** (pixel-level explainability).

## 2. The Core Problem

To classify a WSI ($100,000+$ patches) end-to-end:

1. **Standard End-to-End**: Runs out of GPU memory.
2. **MIL (Multiple Instance Learning)**: Uses frozen features. Fast, but lacks fine-grained feature learning and pixel-level gradients.
3. **IPS (Iterative Patch Selection)**: Selects subsets efficiently but breaks gradient chains if embeddings are cached, preventing CRP.

## 3. The IDPS-CRP Solution

The method employs a **Two-Pass Sparse Re-computation** strategy.

## 3. The IDPS-CRP Solution (Sparse Re-computation)

The architecture follows a **Scout-and-Learn** pattern.

### Phase 1: The Scout (Global Selection)

* **Context**: `torch.no_grad()`
* **Input**:
  * **Option A (Standard)**: All $N$ patches from the WSI.
  * **Option B (Downsampled/Hierarchical)**: Patches extracted from a downsampled level (e.g., 10x). This is the user-requested "Test Mode" to verify if coarse localization suffices.
* **Action**: Feed patches through the *Shared Backbone* (Frozen).
* **Output**: Top-K Indices.
  * *Note*: If Option B is used, indices are mapped to the High-Resolution coordinate space.

### Phase 2: The Learner (Targeted Training)

* **Context**: `requires_grad=True` (This is the "DPS-like" phase).
* **Input**: **High-Resolution Patches** (e.g., 40x) extracted at the specific locations found by the Scout.
  * *Correction*: Even if Phase 1 was downsampled, Phase 2 typically reverts to High-Res for fine-grained subtype classification.
* **Action**:
  1. **Re-Embed**: Pass the $K$ high-res patches through the **SAME Shared Backbone** (now Unfrozen).
  2. **Backpropagate**: Compute loss and update weights.
* **Why this works**: The gradients update the backbone. Because the backbone is *shared*, the "Scout" (Phase 1) implicitly becomes smarter in the next epoch, even though the selection step itself was non-differentiable.

### Addressing the "Re-embed" Question

> *"Can't we just reembed the downsampled version?"*

Technically, yes, you *could* re-embed the downsampled patches in Phase 2, but that would limit the model to coarse features. The power of IDPS-CRP, as visualized in `idps-crp.png` (implied), is:
**Scout on Low-Res (Fast) $\to$ Train on High-Res (Accurate).**
The "Re-computation" step is essential precisely because we switch data sources (from cached/low-res to fresh high-res) OR generically to enable the gradient tape (which was off in Phase 1).

## 4. Explainability (CRP)

Because Pass 2 establishes a fully differentiable path from Prediction to Pixels, we can apply **Concept Relevance Propagation (CRP)**.

* **LRP (Layer-wise Relevance Propagation)**: Decomposes the prediction score into relevance scores for each input pixel.
* **CRP**: Disentangles the explanation by concepts (e.g., "Which pixels contributed to the 'Glandular' concept activation?").

## 5. Implementation Specifications

* **Backbone**: ResNet-18 (truncated) or similar CNN.
* **Aggregator**: Multi-head Self-Attention Transformer.
* **Loss**: Cross-Entropy (for classification).
* **Hyperparameters**:
  * $N$: Total patches (variable, ~10k-100k).
  * $K$: Selected patches (e.g., 64-128).
  * $M$: Chunk size for Pass 1 (e.g., 512).

## 6. Advantages

1. **Constant Training Memory**: Dependent only on $K$, allowing training on consumer GPUs.
2. **End-to-End Feature Learning**: The encoder is not frozen.
3. **Pixel-Level Explanation**: Enables "Why is this patch cancerous?" analysis, not just "This patch is important".
