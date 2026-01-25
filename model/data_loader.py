import os
import numpy as np
import torch


class FastDataLoader:
    def __init__(self, data_path, block_size, batch_size, device):
        """
        Loads data into memory once
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        print(f"Loading {data_path} into memory...")
        
        data_memmap = np.memmap(data_path, dtype=np.uint16, mode='r')
        data_np = np.array(data_memmap, dtype=np.int64)
        
        if device == 'cuda':
            self.data = torch.from_numpy(data_np).to(device)
            print(f"  Loaded {len(data_np):,} tokens to GPU ({len(data_np)*2/1e6:.1f}MB)")
        else:
            self.data = torch.from_numpy(data_np)
            print(f"  Loaded {len(data_np):,} tokens to CPU ({len(data_np)*2/1e6:.1f}MB)")
        
        self.data_size = len(self.data) - block_size - 1
        
        # Pre-allocate index buffer
        self.index_buffer = torch.empty(batch_size, dtype=torch.long, device=device)
        
        self.offsets = torch.arange(block_size, device=device)
    
    def get_batch(self):
        torch.randint(0, self.data_size, (self.batch_size,), out=self.index_buffer)
        
        indices = self.index_buffer.unsqueeze(1)  # (batch_size, 1)
        
        idx_x = indices + self.offsets  # (batch_size, block_size)
        idx_y = idx_x + 1
        
        x = self.data[idx_x]
        y = self.data[idx_y]
        
        return x, y


class ConsolidatedDataLoader:
    """
    Data loader that consolidates multiple files into one dataset.
    Used to consolidate train+val into training set.
    """
    def __init__(self, data_paths, block_size, batch_size, device):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # Load all files and concatenate
        data_arrays = []
        for path in data_paths:
            data_memmap = np.memmap(path, dtype=np.uint16, mode='r')
            data_np = np.array(data_memmap, dtype=np.int64)
            data_arrays.append(data_np)
            print(f"  {os.path.basename(path)}: {len(data_np):,} tokens")
        
        # Concatenate all data
        data_np = np.concatenate(data_arrays)
        print(f"  Total consolidated: {len(data_np):,} tokens ({len(data_np)*2/1e6:.1f}MB)")
        
        if device == 'cuda':
            self.data = torch.from_numpy(data_np).to(device)
        else:
            self.data = torch.from_numpy(data_np)
        
        self.data_size = len(self.data) - block_size - 1
        
        self.index_buffer = torch.empty(batch_size, dtype=torch.long, device=device)
        self.offsets = torch.arange(block_size, device=device)
    
    def get_batch(self):
        torch.randint(0, self.data_size, (self.batch_size,), out=self.index_buffer)
        
        indices = self.index_buffer.unsqueeze(1)
        
        idx_x = indices + self.offsets
        idx_y = idx_x + 1
        
        x = self.data[idx_x]
        y = self.data[idx_y]
        
        return x, y


def get_loaders(data_dir, block_size, batch_size, device, consolidate_train_val=False):
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    test_path = os.path.join(data_dir, 'test.bin')
    
    if consolidate_train_val:
        print("\n" + "="*60)
        print("CONSOLIDATING TRAIN + VAL")
        print("="*60)
        print("Training on: train.bin + val.bin")
        print("Validation on: test.bin")
        print("="*60 + "\n")
        
        train_loader = ConsolidatedDataLoader(
            [train_path, val_path],
            block_size,
            batch_size,
            device
        )
        
        val_loader = FastDataLoader(test_path, block_size, batch_size, device)
        test_loader = val_loader
        
    else:
        print("\n" + "="*60)
        print("STANDARD DATA SPLIT")
        print("="*60)
        print("Training on: train.bin")
        print("Validation on: val.bin")
        print("Test on: test.bin")
        print("="*60 + "\n")
        
        # Standard: separate train/val/test
        train_loader = FastDataLoader(train_path, block_size, batch_size, device)
        val_loader = FastDataLoader(val_path, block_size, batch_size, device)
        test_loader = FastDataLoader(test_path, block_size, batch_size, device)
    
    return train_loader, val_loader, test_loader