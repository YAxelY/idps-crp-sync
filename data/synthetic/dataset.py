import torch
from torch.utils.data import Dataset

class SyntheticWSIDataset(Dataset):
    """
    Simulates a dataset of Whole Slide Images (WSIs).
    Each item is a "Bag" of patches.
    Supports "Scout" (Low-Res) and "Learner" (High-Res) access patterns.
    """
    def __init__(self, num_slides=10, n_patches=1000, low_res_size=32, high_res_size=224, n_classes=2):
        self.num_slides = num_slides
        self.n_patches = n_patches
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.n_classes = n_classes
        
        # Determine downsample factor
        self.downsample_factor = self.high_res_size // self.low_res_size

    def __len__(self):
        return self.num_slides

    def __getitem__(self, idx):
        # In a real scenario, this would load metadata for slide `idx`
        # Here we verify we can return a "slide object" or similar wrapper
        # For simplicity in PyTorch Dataloader, we'll return a dict 
        # that allows downstream collate_fn to handle it.
        
        return {
            'slide_id': idx,
            'label': torch.randint(0, self.n_classes, (1,)).item(),
            'n_patches': self.n_patches
        }

    # --- Methods called by the Trainer/Model directly ---

    def get_scout_data(self, batch_data):
        """
        Returns the Low-Res patches for the Scout Pass.
        batch_data: output of __getitem__ (or collated)
        Returns: (B, N, C, H_low, W_low)
        """
        # In real app: Load images from disk at low magnification (e.g. level 2)
        # Here: Generate random tensors
        B = len(batch_data['slide_id']) if isinstance(batch_data['slide_id'], list) else 1
        return torch.randn(B, self.n_patches, 3, self.low_res_size, self.low_res_size)

    def get_learner_data(self, batch_data, indices):
        """
        Returns the High-Res patches for the Learner Pass.
        batch_data: output of __getitem__
        indices: (B, K) selected indices from Scout Pass
        Returns: (B, K, C, H_high, W_high)
        """
        # In real app: 
        # 1. Map indices to High-Res coordinates (if needed)
        # 2. Extract crops from Level 0 WSI
        
        B, K = indices.shape
        return torch.randn(B, K, 3, self.high_res_size, self.high_res_size)
