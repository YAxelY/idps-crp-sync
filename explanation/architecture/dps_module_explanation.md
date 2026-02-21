# DPS Module Explanation (`dps_module.py`)

This module implements the **Differentiable Patch Selection (DPS)** mechanism using the **Perturbed Top-K** method. It allows the model to select the top-$K$ most relevant patches in a way that is differentiable, enabling end-to-end training with backpropagation.

## Key Mathematical Concept: Perturbed Top-K

Standard Top-K selection is a discrete operation (hard selection), which has zero gradient almost everywhere. To make it differentiable, we smooth the operation by adding noise to the scores.

Given scores $s$, we compute the perturbed scores:
$$ \tilde{s} = s + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

The forward pass computes the indicators based on the perturbed scores. The backward pass estimates the gradients using the noise distribution.

## Class: `PerturbedTopK`

This is the high-level PyTorch Module wrapper.

```python
class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        """
        k: Number of elements to select.
        num_samples: Number of noise samples for gradient estimation (Monte Carlo).
        sigma: Noise level (temperature). Controls the smoothness/exploration.
        """
        ...

    def __call__(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)
```

## Class: `PerturbedTopKFunction` (Autograd Function)

This class defines the custom forward and backward passes.

### Forward Pass

1.  **Noise Generation**: Generates Gaussian noise $\epsilon$.
2.  **Perturbation**: Adds noise to input scores $x$.
    $$ \tilde{x} = x + \sigma \epsilon $$
3.  **Top-K Selection**: Performs hard Top-K on the *perturbed* scores $\tilde{x}$.
4.  **Indicator Creation**: Creates a one-hot matrix indicating selected indices.
5.  **Averaging**: The output `indicators` is the mean of these one-hot matrices across the batch dimension (if using multiple noise samples per input, though here `num_samples` seems to be used for gradient estimation dimension).

*Note: In this implementation, `num_samples` is used to create a noise tensor of shape `(B, num_samples, D)`. The forward pass effectively averages over `num_samples` parallel perturbations to get a logical "soft" indicator.*

```python
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        # ...
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma 
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        # ...
        indicators = perturbed_output.mean(dim=1) 
        # ...
        return indicators
```

### Backward Pass

Uses the **Perturbed Optimizers** gradient estimation trick. The gradient with respect to the input scores $x$ is estimated using the correlation between the output (indicators) and the noise $\epsilon$.

$$ \nabla_x \mathbb{E}[f(x + \sigma \epsilon)] = \frac{1}{\sigma} \mathbb{E}[f(x + \sigma \epsilon) \epsilon^\top] $$

```python
    @staticmethod
    def backward(ctx, grad_output):
        # ...
        # Formula: (PerturbedOutput * Noise) / (NumSamples * Sigma)
        expected_gradient = (
            torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        ) * float(ctx.k)
        
        # Chain rule: grad_output * local_gradient
        grad_input = torch.einsum("bkd,bkde->be", grad_output, expected_gradient)
        
        return (grad_input,) + tuple([None] * 5)
```

This gradient estimation allows the network to learn *which* patches to select to minimize the final loss, bridging the gap between the scorer (Pass 1/Learner) and the aggregator.
