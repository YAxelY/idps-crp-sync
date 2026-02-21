import os
import json
import numpy as np
import torch
import torch.nn.functional as F

class MegapixelMNIST(torch.utils.data.Dataset):
    """ Loads the Megapixel MNIST dataset with true lazy loading """

    def __init__(self, conf, train=True):
        with open(os.path.join(conf.data_dir, "parameters.json")) as f:
            self.parameters = json.load(f)
        
        self.conf = conf
        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks

        filename = "train.npy" if train else "test.npy"
        self.W = self.parameters["width"]
        self.H = self.parameters["height"]
        
        # Load sparse data annotations
        self._data = np.load(os.path.join(conf.data_dir, filename), allow_pickle=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        """ 
        Just returns the raw dictionary with sparse coordinates instead of 
        reconstructing the gigapixel image eagerly in RAM/VRAM.
        """
        if i >= len(self):
            raise IndexError()
        return self._data[i]

    def _reconstruct_dense(self, data):
        """ Reconstruct full image on CPU. Internal helper, avoid calling if possible. """
        img = np.zeros((self.H, self.W, 1), dtype=np.float32).ravel()
        img[data['input'][0]] = data['input'][1]
        img = img.reshape((self.H, self.W, 1))
        img = torch.from_numpy(img).permute(2, 0, 1) # (1, H, W)
        return img

    def get_scout_data(self, batch_data_list):
        """
        Takes a list of raw sparse data dicts (batch).
        Reconstructs the full image on CPU, spatially downsamples it directly 
        (e.g., 1500x1500 -> 300x300), and then extracts the corresponding downsampled patches.
        This provides the Scout phase with global context WITHOUT extracting heavy Nx(1x50x50) tensors.
        Returns: 
           patches: (B, N, C, H_low, W_low)
           labels: dict of stacked tensors
        """
        batch_size = len(batch_data_list)
        B_patches = []
        labels = {task['name']: [] for task in self.tasks.values()}

        for i in range(batch_size):
            data = batch_data_list[i]
            
            # 1. Reconstruct full WSI (1, H, W)
            img = self._reconstruct_dense(data).unsqueeze(0) # (1, 1, H, W)
            
            # 2. Downsample the whole image directly
            # E.g., if downsample_factor=5, 1500x1500 -> 300x300
            # If so, the patch size and stride must also be scaled down equivalently!
            if hasattr(self.conf, 'downsample_factor') and self.conf.downsample_factor > 1:
                df = self.conf.downsample_factor
                img_low = F.interpolate(img, scale_factor=1.0/df, mode='bilinear', align_corners=False)
                p_size = (self.patch_size[0] // df, self.patch_size[1] // df)
                p_stride = (self.patch_stride[0] // df, self.patch_stride[1] // df)
            else:
                img_low = img
                p_size = self.patch_size
                p_stride = self.patch_stride
            
            img_low = img_low.squeeze(0) # (1, H_low, W_low)
            
            # 3. Extract Downsampled Patches
            patches = img_low.unfold(
                1, p_size[0], p_stride[0]
            ).unfold(
                2, p_size[1], p_stride[1]
            ).permute(1, 2, 0, 3, 4)
            patches = patches.reshape(-1, *patches.shape[2:]) # (N, 1, p_size, p_size)
            
            B_patches.append(patches)
            for task in self.tasks.values():
                labels[task['name']].append(torch.tensor(data[task['name']]))
        
        # Stack into batch
        batch_patches = torch.stack(B_patches) # (B, N, 1, p_h, p_w)
        batch_labels = {k: torch.stack(v) for k, v in labels.items()}
        
        return batch_patches, batch_labels

    def get_learner_data(self, batch_data_list, indices):
        """
        Takes the raw sparse batch AND the chosen M indices from Scout.
        Extracts ONLY the requested M high-resolution patches per item in the batch.
        
        indices: (B, M) tensor from Scout (CPU or GPU, will move to CPU to slice)
        Returns:
           selected_patches_high: (B, M, C, H_high, W_high)
        """
        indices = indices.cpu()
        batch_size, M = indices.shape
        B_patches_high = []

        # Find how many patches there are per dimension to map 1D index -> 2D (y, x)
        # N_h = (H - patch_size) // patch_stride + 1
        n_h = (self.H - self.patch_size[0]) // self.patch_stride[0] + 1
        n_w = (self.W - self.patch_size[1]) // self.patch_stride[1] + 1

        for i in range(batch_size):
            data = batch_data_list[i]
            # Since we only need M patches, reconstructing the whole img might still be simpler
            # than advanced sparse querying, but we ONLY unfold and extract M patches.
            img = self._reconstruct_dense(data) # (1, H, W)
            
            m_patches = []
            for m in range(M):
                idx = indices[i, m].item()
                # 1D index to 2D grid
                grid_y = idx // n_w
                grid_x = idx % n_w
                
                # Image pixel coordinates
                start_h = grid_y * self.patch_stride[0]
                start_w = grid_x * self.patch_stride[1]
                end_h = start_h + self.patch_size[0]
                end_w = start_w + self.patch_size[1]
                
                patch = img[:, start_h:end_h, start_w:end_w] # (1, P_h, P_w)
                m_patches.append(patch)
            
            B_patches_high.append(torch.stack(m_patches)) # (M, 1, P_h, P_w)
            
        return torch.stack(B_patches_high) # (B, M, 1, P_h, P_w)