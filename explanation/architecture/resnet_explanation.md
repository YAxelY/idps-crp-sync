# ResNet Explanation (`resnet.py`)

This file contains a standard implementation of the **ResNet** (Residual Network) architecture, with minor modifications to support 1-channel inputs (for MNIST) and flexible flattening.

## Overview
ResNet uses skip connections (residuals) to allow training of very deep networks by mitigating the vanishing gradient problem.
$$ y = F(x) + x $$

## Key Components

### 1. `BasicBlock` vs `Bottleneck`
*   `BasicBlock`: Used for smaller ResNets (ResNet18, 34). Consists of two $3 \times 3$ convolutions.
*   `Bottleneck`: Used for deeper ResNets (ResNet50+). Uses $1\times1 \to 3\times3 \to 1\times1$ structure to reduce dimensionality for the costly $3\times3$ conv.

### 2. `ResNet` Class
The main backbone structure.

**Initialization:**
*   `self.conv1`: Initial $7 \times 7$ convolution.
    *   *Modification*: `num_channels` argument allows adaptation to grayscale (1 channel) vs RGB (3 channels).
*   `self.layer1` to `self.layer4`: The four stages of ResNet, progressively decreasing spatial resolution and increasing channel depth (64 -> 128 -> 256 -> 512).

**Forward Pass (`_forward_impl`):**
1.  **Stem**: Conv1 -> BN -> ReLU -> MaxPool.
2.  **Groups**: Layer1 -> Layer2 -> Layer3 -> Layer4.
3.  **Head (Optional Flattening)**:
    ```python
        if self.flatten:
            x = self.avgpool(x)       # Global Average Pooling (1x1 special)
            x = torch.flatten(x, 1)   # Flatten to (B, 512)
    ```
    IDPSNet uses `flatten=True` to get a feature vector of size 512 for each patch.

### 3. Factory Functions
Functions like `resnet18()` provide a convenient way to instantiate standard configurations.

```python
def resnet18(pretrained: bool = False, ...):
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], ...)
```
The list `[2, 2, 2, 2]` specifies how many residual blocks are in each of the 4 layers.
