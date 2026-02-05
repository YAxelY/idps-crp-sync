# Product Requirements Document (PRD): IDPS-CRP Implementation

## 1. Introduction
This project aims to implement the **IDPS-CRP** architecture for lung cancer subtype classification on Whole Slide Images (WSI). The system provides memory-efficient training and fine-grained interpretability.

## 2. Epics

### Epic 1: Core Architecture Implementation
Establish the `IDPSNet` model with the two-pass mechanism.
*   **Story 1.1**: Implement `Encoder` module (ResNet-based) adaptable for both `eval` and `train` modes.
### Epic 1: Core Architecture (Scout-and-Learn)
*   **Story 1.1**: Implement `IDPSNet` with shared backbone logic.
*   **Story 1.2 (The Scout)**: Implement `scouting_pass(wsi, downsample=False)`.
    *   If `downsample=True`, input is Low-Res.
    *   Returns `top_k_indices`.
*   **Story 1.3 (The Learner)**: Implement `training_pass(wsi, indices)`.
    *   Input: `indices` from Scout.
    *   Action: Extract **High-Res** patches at these indices.
    *   Run Forward Pass with `requires_grad=True`.
*   **Story 1.4**: Implement `IndexMapper` to convert Low-Res indices $\to$ High-Res pixel coordinates.

### Epic 2: Data Pipeline
*   **Story 2.1**: Implement `WSIWrapper` that abstraction that can serve both Low-Res views (for Scout) and High-Res patches (for Learner).
    *   *Critical*: Must handle "on-the-fly" downsampling or load pre-downsampled tensors.

### Epic 3: Explainability (CRP) Integration
Enable interpretability.
*   **Story 3.1**: Instrument the model for CRP/LRP (ensure layers are compatible with `zennit` or custom rules).
*   **Story 3.2**: Create a `visualize_concepts` utility to generate heatmaps for selected patches.

### Epic 4: Training & Evaluation
*   **Story 4.1**: Implement the training loop (Epochs: Select -> Train).
*   **Story 4.2**: Implement logging (Selection stability, Loss, Accuracy).

## 3. Functional Requirements
*   **FR-01**: Model MUST accept a bag of patches (or WSI path) and return a classification score.
*   **FR-02**: Training memory usage MUST NOT exceed 12GB (or configured limit) regardless of WSI size.
*   **FR-03**: The system MUST produce visualization of "relevant concepts" on the input images upon request.

## 4. Technical Constraints
*   **Framework**: PyTorch.
*   **Base Code**: Derive structure from `code-references/ips`.
*   **Target Directory**: `my-code01` (inside workspace).
