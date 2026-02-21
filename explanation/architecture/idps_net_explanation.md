# IDPSNet Explanation (`idps_net.py`)

`IDPSNet` is the central class that orchestrates the **Two-Pass Framework**:
1.  **Scout (Pass 1)**: Global search / filtering.
2.  **Learner (Pass 2)**: Fine-grained selection and classification using DPS.

## Architecture Composition (`__init__`)

*   **Encoder**: A `resnet18` backbone. It processes patches into feature vectors.
    *   `resnet18(..., flatten=True)` $\rightarrow$ Output: 512D vector per patch.
*   **Projector**: A linear layer mapping ResNet features (512D) to the internal dimension $D$ (e.g., 128D).
    *   `self.projector = nn.Linear(512, self.D)`
*   **Scorer / Aggregator**: A `Transformer` (from `transformer.py`).
    *   Used in Pass 1 to find regions of interest.
    *   Used in Pass 2 to aggregate selected patches.
*   **Selector**: The `PerturbedTopK` module (from `dps_module.py`).
    *   Responsible for the differentiable selection of $K$ patches.
*   **Output Heads**: A dictionary of linear layers for classification tasks (e.g., 'majority', 'max').

---

## The Workflow Methods

### 1. `_get_embeddings(patches)`
Helper function to extract features.
*   Input: Patches $(B, N, C, H, W)$ or $(N, C, H, W)$.
*   Operation: `Encoder` $\rightarrow$ `Projector`.
*   Output: Embeddings $(B, N, D)$.

### 2. `scouting_pass(wsi_patches)` (Pass 1)
Designed to run on **Low Resolution** or **Sub-sampled** data to quickly find relevant areas.

1.  **Embed**: Extracts features from the `wsi_patches`.
2.  **Score**: Uses `self.transf.get_scores(embeddings)`. Ideally, the Transformer learns to attend to "abnormal" or "relevant" areas.
3.  **Hard Selection**: Uses `torch.topk` (non-differentiable) to pick top $M$ candidates.
    *   We don't need gradients here because this pass is just for filtering. Only the *embeddings* training is shared.

```python
    top_m_indices = torch.topk(scores, self.M, dim=-1)[1]
```

### 3. `training_pass(candidate_patches)` (Pass 2)
The core **Learner** step. Runs on High-Resolution patches selected by the Scout.

1.  **Embed**: Re-computes embeddings for the $M$ candidates. (Gradients are active here).
2.  **Score**: Computes importance scores via Transformer Attention again.
3.  **Normalization**:
    *   Scores are Min-Max normalized to $[0, 1]$. This is crucial for stability with DPS noise.
4.  **DPS Selection (Differentiable)**:
    *   `indicators = self.dps_topk(scores)`
    *   Result: A fuzzy (soft) one-hot matrix of shape $(B, K, M)$.
5.  **Soft Gather**:
    *   `selected = indicators @ embeddings`
    *   Extracts the $K$ relevant embeddings as a weighted sum.
6.  **Aggregate**:
    *   Passes selected embeddings through the Transformer to get a slide-level vector.
7.  **Classify**:
    *   Passes the vector to task-specific heads to get final predictions.

### 4. `forward(x, mode)`
A switch wrapper. Note that `IDPSNet` is rarely called with `net(x)`. Instead, the training loop calls `net.scouting_pass` and `net.training_pass` explicitly for full control over the two-stage data flow.
