import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import math

from model import GPTConfig, GPT
from data_loader import get_loaders
from base import GPT as Baseline

# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'resume'  # 'scratch', 'resume', or 'gpt2*'

# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# data
dataset = 'enwik8'
gradient_accumulation_steps = 8
batch_size = 6
block_size = 1024
consolidate_train_val = True # If True: train on train+val, validate on test

# model
n_layer = 48
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# adamw optimizer
learning_rate = 6e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 0
lr_decay_iters = 10000
min_lr = 6e-5

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# data loading mode
data_mode = 'random'

# -----------------------------------------------------------------------------

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Setup
master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)  # For loss plots
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Initialize data loaders with optional consolidation
data_dir = os.path.join('../data', dataset)

train_loader, val_loader, test_loader = get_loaders(
    data_dir,
    block_size,
    batch_size,
    device,
    consolidate_train_val=consolidate_train_val
)

print('Initialized dataloaders.')

# -----------------------------------------------------------------------------
# Model initialization
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
else:
    print("Using default vocab_size=256 for enwik8 dataset")
    meta_vocab_size = 256

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 256
    print(f"Using vocab_size = {model_args['vocab_size']} for enwik8 dataset")
    gptconf = GPTConfig(**model_args)
    model = Baseline(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'base_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = Baseline(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = Baseline.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Unknown init_from value: {init_from}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# Initialize GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Evaluation function
@torch.no_grad()
def estimate_loss():
    """Evaluation with fast data loading."""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        losses = torch.zeros(eval_iters, device=device)
        
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss
        
        out[split] = losses.mean().item()
    
    model.train()
    return out


# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# Logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Loss tracking and plotting
train_losses = []
train_steps = []
val_losses = []
val_steps = []

def plot_losses():
    """Plot and save train/val loss curves."""
    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, label='Train', alpha=0.7, linewidth=0.5)
    if val_losses:
        val_label = 'Test' if consolidate_train_val else 'Val'
        plt.plot(val_steps, val_losses, label=val_label, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss (bpc)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plots', f'loss_step_{iter_num}.png'), dpi=100)
    plt.close()

# -----------------------------------------------------------------------------
# Training loop
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Compile enabled: {compile}")
print(f"Consolidate train+val: {consolidate_train_val}")
print("="*60 + "\n")

# Pre-fetch first batch
X, Y = train_loader.get_batch()

t0 = time.time()
local_iter_num = 0
raw_model = model.module if hasattr(model, 'module') else model

while True:
    # Learning rate scheduling
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        val_label = 'test' if consolidate_train_val else 'val'
        print(f"step {iter_num}: train loss {losses['train']/math.log(2):.4f}, {val_label} loss {losses['val']/math.log(2):.4f}")
        
        # Record losses
        val_losses.append(losses['val']/math.log(2))
        val_steps.append(iter_num)
        
        # Plot losses
        plot_losses()
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'base_ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # Gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        
        # Prefetch next batch (now fast!)
        X, Y = train_loader.get_batch()
        
        # Backward pass
        scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        train_losses.append(lossf/math.log(2))
        train_steps.append(iter_num)
        print(f"iter {iter_num}: loss {lossf/math.log(2):.4f}, time {dt*1000:.2f}ms")
    
    iter_num += 1
    local_iter_num += 1

    # Termination
    if iter_num > max_iters:
        break

print("Training completed!")