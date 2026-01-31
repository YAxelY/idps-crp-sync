import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .generator import MegapixelGenerator

class MegapixelMNISTDataset(Dataset):
    def __init__(self, data_dir, train=True, patch_size=50, stride=50):
        self.data_dir = data_dir
        
        # Check if data exists, if not generate
        train_file = os.path.join(data_dir, "train.npy")
        test_file = os.path.join(data_dir, "test.npy")
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Data not found in {data_dir}. Generating on the fly...")
            os.makedirs(data_dir, exist_ok=True)
            # Default generation params (matches config typically)
            # H=500, W=500 for test speed
            gen = MegapixelGenerator(data_dir=data_dir, W=500, H=500) 
            gen.generate_dataset(100, train_file, 'train') # 100 train samples
            gen.generate_dataset(20, test_file, 'test')   # 20 test samples
            
            # Save params
            with open(os.path.join(data_dir, "params.json"), 'w') as f:
                json.dump({'W': 500, 'H': 500, 'patch_size': patch_size}, f)
                
        with open(os.path.join(data_dir, "params.json")) as f:
            self.params = json.load(f)
        
        self.W = self.params['W']
        self.H = self.params['H']
        self.patch_size = patch_size
        self.stride = stride
        
        filename = "train.npy" if train else "test.npy"
        self.data = np.load(os.path.join(data_dir, filename), allow_pickle=True)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Reconstruct sparse -> dense
        img = np.zeros((self.H, self.W), dtype=np.float32)
        img[item['indices']] = item['values']
        
        img_tensor = torch.from_numpy(img).unsqueeze(0) # (1, H, W)
        label = item['label']
        
        # Unfold to patches
        # (1, H, W) -> (1, N_h, N_w, P, P)
        patches = img_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        N_h, N_w = patches.shape[1], patches.shape[2]
        patches = patches.contiguous().view(-1, 1, self.patch_size, self.patch_size) # (N, 1, P, P)
        
        # Repeat channels to match ResNet 3 channels
        patches = patches.repeat(1, 3, 1, 1) # (N, 3, P, P)
        
        return {
            'patches': patches, # (N, 3, P, P)
            'label': label,
            'id': idx
        }

    # --- IDPS Interface ---

    def get_scout_data(self, batch_data):
        """
        Returns all patches for scanning.
        In real WSI, this might return a downsampled view.
        Here, we return the full set of patches.
        """
        # batch_data['patches'] is (B, N, 3, P, P) (collated)
        return batch_data['patches']

    def get_learner_data(self, batch_data, indices):
        """
        Returns specific patches based on scout indices.
        batch_data['patches']: (B, N, 3, P, P)
        indices: (B, M)
        """
        patches = batch_data['patches']
        B, M = indices.shape
        C, H, W = patches.shape[2:]
        
        # Gather (B, M, C, H, W)
        # indices needs expansion: (B, M, 1, 1, 1) to match patch dims?
        # torch.gather is usually for dim 1.
        
        # Standard gather pattern:
        flat_indices = indices.view(B, M, 1, 1, 1).expand(-1, -1, C, H, W)
        selected = torch.gather(patches, 1, flat_indices)
        
        return selected

