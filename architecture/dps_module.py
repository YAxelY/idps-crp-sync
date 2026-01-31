import torch
from torch import nn

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
    
    def __call__(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        b, d = x.shape
        # Input x is typically scores (B, N)
        
        # Generates noise
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        # Add noise
        perturbed_x = x[:, None, :] + noise * sigma # (B, S, N)
        
        # Top-K on perturbed
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # (B, S, K)
        indices = torch.sort(indices, dim=-1).values # (B, S, K)

        # One-hot
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        
        # Average indicators (B, K, N)
        indicators = perturbed_output.mean(dim=1) 
        
        # Context for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        ) * float(ctx.k)
        
        grad_input = torch.einsum("bkd,bkde->be", grad_output, expected_gradient)
        
        return (grad_input,) + tuple([None] * 5)
