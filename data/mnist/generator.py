import os
import argparse
import json
import numpy as np

import torchvision
from torchvision import datasets, transforms

# Adapted from reference make_mnist.py
# Using torch datasets instead of assuming keras/files present

class MegapixelGenerator:
    def __init__(self, data_dir, W=1500, H=1500, n_noise=50, noise=True):
        self.W = W
        self.H = H
        self.n_noise = n_noise
        self.noise = noise
        self.data_dir = data_dir
        
        # Load MNIST (downloads if needed)
        # Store raw mnist in data_dir/raw_mnist to keep it clean
        raw_dir = os.path.join(data_dir, 'raw_mnist')
        os.makedirs(raw_dir, exist_ok=True)
        
        self.mnist_train = datasets.MNIST(raw_dir, train=True, download=True, 
                                        transform=transforms.ToTensor())
        self.mnist_test = datasets.MNIST(raw_dir, train=False, download=True,
                                       transform=transforms.ToTensor())

        self.train_data = self.mnist_train.data.numpy()
        self.train_targets = self.mnist_train.targets.numpy()
        
        self.test_data = self.mnist_test.data.numpy()
        self.test_targets = self.mnist_test.targets.numpy()

    def create_sample(self, dataset_type='train'):
        if dataset_type == 'train':
            X, y = self.train_data, self.train_targets
        else:
            X, y = self.test_data, self.test_targets
            
        # 1. Select 5 digits
        # Target digit (random)
        target_digit = np.random.randint(0, 10)
        
        # 3 Positive instances (same digit)
        pos_indices = np.random.choice(np.where(y == target_digit)[0], 3)
        
        # 2 Negative instances (diff digit)
        neg_indices = np.random.choice(np.where(y != target_digit)[0], 2)
        
        all_indices = np.concatenate([pos_indices, neg_indices])
        np.random.shuffle(all_indices)
        
        # 2. Place them randomly
        canvas = np.zeros((self.H, self.W), dtype=np.float32)
        
        for idx in all_indices:
            digit = X[idx] # 28x28
            # random pos
            r = np.random.randint(0, self.H - 28)
            c = np.random.randint(0, self.W - 28)
            canvas[r:r+28, c:c+28] = np.maximum(canvas[r:r+28, c:c+28], digit)

        if self.noise:
             # Add noise spots
             for _ in range(self.n_noise):
                 rx = np.random.randint(0, self.H - 28)
                 cx = np.random.randint(0, self.W - 28)
                 noise_patch = np.random.rand(28, 28) * 255 * 0.5
                 canvas[rx:rx+28, cx:cx+28] = np.maximum(canvas[rx:rx+28, cx:cx+28], noise_patch)

        # Normalize
        canvas = canvas / 255.0
        
        return canvas, target_digit

    def generate_dataset(self, n_samples, output_path, dataset_type='train'):
        data_list = []
        print(f"Generating {n_samples} {dataset_type} samples...")
        for i in range(n_samples):
            img, label = self.create_sample(dataset_type)
            # Sparse save
            indices = np.where(img > 0.1) 
            values = img[indices] # simple threshold sparse
            
            data_list.append({
                'indices': indices,
                'values': values,
                'label': label
            })
        
        np.save(output_path, data_list)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="my-code01/data/mnist/data")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=20)
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    gen = MegapixelGenerator(W=500, H=500) # Small for test
    gen.generate_dataset(args.n_train, os.path.join(args.out, "train.npy"), 'train')
    gen.generate_dataset(args.n_test, os.path.join(args.out, "test.npy"), 'test')
    
    # Save params
    with open(os.path.join(args.out, "params.json"), 'w') as f:
        json.dump({'W': 500, 'H': 500, 'patch_size': 50}, f)
