import os
import argparse
import torch
from torch.utils.data import DataLoader
from architecture.idps_net import IDPSNet
from training.trainer import IDPSTrainer
from config.loader import load_config
from data.synthetic.dataset import SyntheticWSIDataset
from data.mnist.dataset import MegapixelMNISTDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mnist', choices=['mnist', 'synthetic'], 
                        help='Task to run: mnist or synthetic')
    return parser.parse_args()

def main():
    args = get_args()
    task = args.task
    
    # Load Config
    config_path = f'my-code01/config/{task}_config.yaml'
    print(f"Loading config from {config_path}")
    conf = load_config(config_path)
    
    # Update device
    if isinstance(conf.device, str):
        if conf.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            conf.device = 'cpu'
        conf.device = torch.device(conf.device)
    print(f"Running on {conf.device}")
    
    # Init Data
    print(f"Initializing {task} dataset...")
    if task == 'synthetic':
        # Synthetic is random, so we just instantiate twice
        train_dataset = SyntheticWSIDataset(
            num_slides=conf.num_slides, n_patches=conf.n_patches,
            low_res_size=conf.low_res_size, high_res_size=conf.high_res_size)
        test_dataset = SyntheticWSIDataset(
            num_slides=conf.num_slides, n_patches=conf.n_patches,
            low_res_size=conf.low_res_size, high_res_size=conf.high_res_size)
        collate_fn = None
    elif task == 'mnist':
        train_dataset = MegapixelMNISTDataset(data_dir=conf.data_dir, train=True, patch_size=conf.patch_size, stride=conf.stride)
        test_dataset = MegapixelMNISTDataset(data_dir=conf.data_dir, train=False, patch_size=conf.patch_size, stride=conf.stride)
        collate_fn = None 

    batch_size = getattr(conf, 'batch_size', 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Init Model
    # Conf should have M (Candidates) and K (Selected)
    model = IDPSNet(conf).to(conf.device)
    
    # Init Trainer
    # Pass train dataset for now, but trainer handles batch data directly mostly
    trainer = IDPSTrainer(model, train_dataset, conf)
    
    # Train
    print("Starting Training & Testing...")
    for epoch in range(conf.n_epoch):
        # 1. Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch+1} [TRAIN]: Loss = {train_metrics['loss']:.4f}, Acc = {train_metrics['acc']:.4f}, Time = {train_metrics['time_sec']:.2f}s, VRAM = {train_metrics['vram_gb']:.2f} GB")
        
        # 2. Test
        test_metrics = trainer.evaluate(test_loader, epoch)
        print(f"Epoch {epoch+1} [TEST ]: Loss = {test_metrics['loss']:.4f}, Acc = {test_metrics['acc']:.4f}, Time = {test_metrics['time_sec']:.2f}s, VRAM = {test_metrics['vram_gb']:.2f} GB")
    
    print("Training Complete.")

if __name__ == "__main__":
    main()
