# Utilities Explanation (`utils.py`)

This file contains helper functions for training schedules, logging, and metrics.

## 1. Schedulers

### `adjust_learning_rate`
Implements a **Cosine Annealing** schedule with **Linear Warmup**.

$$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t - T_{warmup}}{T_{max}} \pi)) $$

*   **Warmup**: Linearly increases LR from 0 to `max_lr` for the first few epochs. This prevents divergence at the start when weights are random.
*   **Decay**: Smoothly decreases LR to near zero.

### `adjust_sigma`
**Proprietary Scheduler for DPS**.
Controls the `sigma` (noise level) parameter in the `PerturbedTopK` module.

*   **Logic**:
    *   High Sigma (Start): More noise $\to$ Gradients flow to more patches (Exploration). The gradients are "smoother" but higher variance.
    *   Low Sigma (End): Less noise $\to$ Selection approximates hard Top-K (Exploitation).
*   It updates `net.dps_topk.sigma` directly in the training loop.

## 2. `Logger` Class
A robust logging utility to track training progress.

*   **`update`**: Accumulates losses and predictions for a batch.
*   **`compute_metric`**: Calculates epoch-level metrics based on the task type:
    *   **Accuracy**: Simple correct/total.
    *   **Multilabel Accuracy**: Used if one input can have multiple tags.
    *   **AUC (Area Under Curve)**: Useful for imbalanced datasets (cancer vs non-cancer).
*   **`print_stats`**: Formats and prints the metrics to the console.

## 3. Helper Functions

### `Struct`
A simple class to convert a dictionary into an object with attributes (e.g., `conf['lr']` becomes `conf.lr`).
```python
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
```

### `get_gpu_memory`
Wrapper around `torch.cuda.memory_allocated()` to get current VRAM usage in GB. Used for the efficiency benchmarks.
