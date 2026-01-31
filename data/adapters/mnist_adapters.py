"""
Adapter datasets for DPS and IPS to use the shared Megapixel MNIST data.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

# Add my-code01 to path to import our generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.mnist.generator import MegapixelGenerator

class MegapixelMNISTForDPS(Dataset):
    """
    Adapter for DPS model. Returns (img_high, img_low, label).
    DPS expects high_size and low_size images, with 3 channels.
    """
    def __init__(self, data_dir, high_size=(500, 500), low_size=(100, 100), train=True):
        self.data_dir = data_dir
        self.high_size = high_size
        self.low_size = low_size
        
        # Check if data exists
        train_file = os.path.join(data_dir, "train.npy")
        test_file = os.path.join(data_dir, "test.npy")
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Data not found in {data_dir}. Generating...")
            os.makedirs(data_dir, exist_ok=True)
            gen = MegapixelGenerator(data_dir=data_dir, W=high_size[1], H=high_size[0])
            gen.generate_dataset(100, train_file, 'train')
            gen.generate_dataset(20, test_file, 'test')
            with open(os.path.join(data_dir, "params.json"), 'w') as f:
                json.dump({'W': high_size[1], 'H': high_size[0]}, f)
        
        with open(os.path.join(data_dir, "params.json")) as f:
            self.params = json.load(f)
        
        filename = "train.npy" if train else "test.npy"
        self._data = np.load(os.path.join(data_dir, filename), allow_pickle=True)
        self.W = self.params['W']
        self.H = self.params['H']
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        item = self._data[idx]
        indices = item['indices']
        values = item['values']
        label = item['label']
        
        # Reconstruct dense image
        img = np.zeros((self.H, self.W), dtype=np.float32)
        img[indices] = values
        
        # Resize to high_size using interpolation
        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        img_high = pil_img.resize((self.high_size[1], self.high_size[0]), Image.BILINEAR)
        img_low = pil_img.resize((self.low_size[1], self.low_size[0]), Image.BILINEAR)
        
        # To tensor and normalize (3 channels for ResNet)
        img_high = np.array(img_high, dtype=np.float32) / 255.0
        img_low = np.array(img_low, dtype=np.float32) / 255.0
        
        # Stack to 3 channels
        img_high = np.stack([img_high, img_high, img_high], axis=0)
        img_low = np.stack([img_low, img_low, img_low], axis=0)
        
        return torch.tensor(img_high), torch.tensor(img_low), label


class MegapixelMNISTForIPS(Dataset):
    """
    Adapter for IPS model. Returns {'input': patches, 'cls': label}.
    IPS expects pre-extracted patches.
    """
    def __init__(self, conf, train=True):
        self.data_dir = conf.data_dir
        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks
        
        # Check if data exists
        train_file = os.path.join(self.data_dir, "train.npy")
        test_file = os.path.join(self.data_dir, "test.npy")
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Data not found in {self.data_dir}. Generating...")
            os.makedirs(self.data_dir, exist_ok=True)
            gen = MegapixelGenerator(data_dir=self.data_dir, W=conf.W, H=conf.H)
            gen.generate_dataset(100, train_file, 'train')
            gen.generate_dataset(20, test_file, 'test')
            with open(os.path.join(self.data_dir, "params.json"), 'w') as f:
                json.dump({'W': conf.W, 'H': conf.H, 'width': conf.W, 'height': conf.H}, f)
        
        with open(os.path.join(self.data_dir, "parameters.json" if os.path.exists(os.path.join(self.data_dir, "parameters.json")) else "params.json")) as f:
            self.parameters = json.load(f)
        
        self.W = self.parameters.get('width', self.parameters.get('W'))
        self.H = self.parameters.get('height', self.parameters.get('H'))
        self._img_shape = (self.H, self.W, 1)
        
        filename = "train.npy" if train else "test.npy"
        self._data = np.load(os.path.join(self.data_dir, filename), allow_pickle=True)
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        item = self._data[idx]
        indices = item['indices']
        values = item['values']
        label = item['label']
        
        # Reconstruct dense image
        img = np.zeros(self._img_shape, dtype=np.float32).ravel()
        # Flatten indices
        flat_indices = indices[0] * self.W + indices[1]
        img[flat_indices] = values
        img = img.reshape(self._img_shape)
        
        img = torch.from_numpy(img).permute(2, 0, 1)  # (1, H, W)
        
        # Extract patches
        patches = img.unfold(1, self.patch_size[0], self.patch_stride[0]).unfold(2, self.patch_size[1], self.patch_stride[1])
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 1, self.patch_size[0], self.patch_size[1])
        
        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = label
        
        return data_dict
