import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from types import SimpleNamespace

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data.megapixel_mnist.make_mnist import MegapixelMNIST as MakerMNIST
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST as LoaderMNIST

def visualize_data():
    """
    1. Generates a small sample of data using make_mnist logic (or loads existing).
    2. Inspects the sparse structure.
    3. Uses Loader to reconstruct and visualize.
    """
    print("--- 1. Data Generation Inspection ---")
    # Simulate params
    W, H = 1500, 1500
    N = 2
    
    # We can try to generate on the fly using the class from make_mnist
    # But make_mnist.MegapixelMNIST __init__ triggers a full generation from keras.
    # We will just assume data exists or generate a small one.
    
    output_dir = "results/test_vis_mnist"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run make_mnist generation for a tiny set
    print(f"Generating tiny dataset in {output_dir}...")
    import subprocess
    
    # Calculate path to make_mnist.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script is in idps-crp-sync/test/data
    # make_mnist is in idps-crp-sync/data/megapixel_mnist
    make_mnist_path = os.path.abspath(os.path.join(script_dir, '../../data/megapixel_mnist/make_mnist.py'))
    
    cmd = [
        "python", make_mnist_path,
        "--n_train", "5", "--n_test", "2",
        "--width", str(W), "--height", str(H),
        "--n_noise", "10",
        output_dir
    ]
    subprocess.check_call(cmd)
    
    print("\n--- 2. Sparse File Inspection ---")
    data_path = os.path.join(output_dir, "train.npy")
    raw_data = np.load(data_path, allow_pickle=True)
    
    sample_idx = 0
    sample = raw_data[sample_idx]
    print(f"Loaded sample {sample_idx} from {data_path}")
    print("Keys:", sample.keys())
    print(f"Labels -> Majority: {sample['majority']}, Max: {sample['max']}, Top: {sample['top']}")
    
    # Inspect Sparse Format
    indices, values = sample['input']
    
    # Handle indices as tuple (from np.where)
    indices_arr = indices[0] if isinstance(indices, tuple) else indices
    
    print(f"Sparse Input: Indices shape {indices_arr.shape}, Values shape {values.shape}")
    print(f"Sparsity ratio: {len(values) / (W*H):.6f} (Non-zero pixels / Total)")
    
    print("\n--- 3. Dataset Loader Reconstruction ---")
    # Mock config
    conf = SimpleNamespace(
        data_dir=output_dir,
        patch_size=(50, 50),
        patch_stride=(50, 50),
        downsample=False,
        tasks={'majority': {'name': 'majority'}}
    )
    
    loader = LoaderMNIST(conf, train=True)
    print(f"Loader initialized. Length: {len(loader)}")
    
    # Get Item
    item = loader[sample_idx]
    patches = item['input'] # (N_patches, C, H, W) -> flattened patches actually in loader?
    # Loader returns: patches = patches.reshape(-1, *patches.shape[2:]) -> (N, C, 50, 50)
    
    print(f"Loader Output 'input' shape: {patches.shape}")
    
    # Reconstruct full image for visualization from patches (approx) or just manual reconstruction from sparse
    # Manual reconstruction from sparse to show "What the model sees"
    img_flat = np.zeros((H*W,), dtype=np.float32)
    img_flat[indices[0]] = values # indices is tuple (array,)
    img_reconstructed = img_flat.reshape(H, W)
    
    print("Saving visualization to 'mnist_sample_vis.png'...")
    plt.figure(figsize=(10, 10))
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title(f"Reconstructed 1500x1500px\nLabels: Maj={sample['majority']}, Top={sample['top']}")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "mnist_sample_vis.png"))
    print("Done.")

if __name__ == "__main__":
    visualize_data()
