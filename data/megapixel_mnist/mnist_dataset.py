import os
import json
import numpy as np
import torch
import torch.nn.functional as F

class MegapixelMNIST(torch.utils.data.Dataset):
    """ Loads the Megapixel MNIST dataset """

    def __init__(self, conf, train=True):
        with open(os.path.join(conf.data_dir, "parameters.json")) as f:
            self.parameters = json.load(f)
        
        self.conf = conf

        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks

        filename = "train.npy" if train else "test.npy"
        W = self.parameters["width"]
        H = self.parameters["height"]

        self._img_shape = (H, W, 1)
        self._data = np.load(os.path.join(conf.data_dir, filename), allow_pickle=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()

        patch_size = self.patch_size
        patch_stride = self.patch_stride

        # Placeholders
        img = np.zeros(self._img_shape, dtype=np.float32).ravel()

        # Fill the sparse representations
        data = self._data[i]
        img[data['input'][0]] = data['input'][1]

        # Reshape to final shape        
        img = img.reshape(self._img_shape)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        # Extract patches
        patches = img.unfold(
            1, patch_size[0], patch_stride[0]
        ).unfold(
            2, patch_size[1], patch_stride[1]
        ).permute(1, 2, 0, 3, 4)
        
        patches = patches.reshape(-1, *patches.shape[2:])

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = data[task['name']] 

        return data_dict

    def get_scout_data(self, batch_data):
        """
        Returns the data for the 'Scout' (Pass 1).
        If downsample is enabled, interpolates the patches.
        """
        patches = batch_data['input'] # (B, N, C, H, W) or (B, N, H, W, C) depending on loading
        # Patches are already standard tensors due to collate or init? 
        # Actually in __getitem__ they are permuted to (1, 2, 0, 3, 4) then reshaped
        # patches shape here from loader: (B, N, C, H, W)
        
        if self.conf.downsample:
            # Flatten to (B*N, C, H, W) for interpolation
            B, N, C, H, W = patches.shape
            patches_flat = patches.view(-1, C, H, W)
            
            # Downsample
            # e.g. 50x50 -> 12x12
            new_size = (H // self.conf.downsample_factor, W // self.conf.downsample_factor)
            
            patches_small = F.interpolate(patches_flat, size=new_size, mode='bilinear', align_corners=False)
            
            # Reshape back
            patches = patches_small.view(B, N, C, new_size[0], new_size[1])
            
        return patches

    def get_learner_data(self, batch_data, indices):
        """
        Returns the High-Res data for the 'Learner' (Pass 2).
        indices: (B, M) - Indices of patches to select
        """
        input_patches = batch_data['input'] # (B, N, C, H, W)
        
        # Gather patches
        # indices are likely on GPU, input_patches might be CPU if lazy, but here likely GPU due to training loop
        # We need to gather strictly.
        
        B, M = indices.shape
        B_in, N, C, H, W = input_patches.shape
        
        # We need to select specific patches.
        # Simple for loop for clarity or gather magic.
        # advanced indexing: Input[i, indices[i]]
        
        # Create batch indices [0, 1, ... B-1] repeated M times -> (B, M)
        batch_idx = torch.arange(B, device=indices.device).unsqueeze(1).expand(B, M)
        
        selected_patches = input_patches[batch_idx, indices] # (B, M, C, H, W)
        
        return selected_patches