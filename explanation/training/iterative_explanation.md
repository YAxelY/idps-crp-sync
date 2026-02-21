# Training Loop Explanation (`iterative.py`)

This file contains the core logic for training and evaluating the IDPSNet. It implements the **Two-Pass** strategy (Scout then Learner) within the epoch loop.

## 1. `compute_loss`

Calculates the loss for multiple tasks (e.g., classification, segmentation) and aggregates them.

*   **Inputs**:
    *   `preds`: Dictionary of predictions from the network.
    *   `criterions`: Dictionary of loss functions (e.g., NLLLoss, BCELoss).
    *   `labels`: Ground truth labels.
*   **Process**:
    *   Iterates over each task defined in `conf.tasks`.
    *   Computes `task_loss = criterion(pred, label)`.
    *   Sums up all task losses: `loss = sum(task_losses) / n_tasks`.
*   **Returns**: The scalar `loss` for backpropagation and a dictionary of metrics for logging.

## 2. `train_one_epoch`

Trains the network for a single epoch. This is where the **IDPS (Scout-Learner)** workflow is explicitly executed.

### The Loop Structure

```python
for data_it, data in enumerate(data_loader):
```
The dataloader yields a batch of dictionary-like objects containing the image data (or pointers to it).

### Step 1: Variable Adjustment
Before every step, we adjust the hyperparameters to follow a schedule.
*   **Learning Rate**: `adjust_learning_rate(...)` (Cosine decay).
*   **DPS Sigma**: `adjust_sigma(...)` (Cosine decay).
    *   *Crucial*: Reducing `sigma` over time shifts the DPS selection from "Exploration" (noisy, random) to "Exploitation" (deterministic, greedy).

### Step 2: Pass 1 - The Scout
```python
# --- PASS 1: SCOUT ---
low_res_patches = train_data.get_scout_data(data).to(device)
top_m_indices = net.scouting_pass(low_res_patches) # (B, M)
```
*   **Data**: Fetches low-resolution or downsampled version of the slide.
*   **Net**: Calls `net.scouting_pass` (No Gradients for selection indices).
*   **Result**: Indices of the $M$ most promising patches.

### Step 3: Pass 2 - The Learner
```python
# --- PASS 2: LEARNER (With DPS) ---
top_m_indices_cpu = top_m_indices.cpu()
candidate_patches = train_data.get_learner_data(data, top_m_indices_cpu).to(device)

# Forward (returns dict of predictions)
preds = net.training_pass(candidate_patches)
```
*   **Data**: Fetches High-Resolution patches for *only* the $M$ indices found by the Scout.
*   **Net**: Calls `net.training_pass`.
    *   This includes the Differentiable Patch Selection (DPS) to narrow down from $M$ to $K$.
    *   Gradients flows through this step.

### Step 4: Optimization
```python
loss, task_info = compute_loss(...)
loss.backward()
optimizer.step()
```
Standard PyTorch optimization step.

### Efficiency Tracking
The code includes `torch.cuda.Event` based timing and `torch.cuda.memory_stats()` to rigorously profile the training speed and VRAM usage, which is part of the thesis goals.

## 3. `evaluate`

Similar to `train_one_epoch` but with key differences:
*   `@torch.no_grad()`: Disables gradient tracking to save memory.
*   `net.eval()`: Switches model to evaluation mode (affects Dropout, BatchNorm).
*   Same Two-Pass logic (Scout -> Learner) to ensure the evaluation metric reflects the actual model pipeline.
