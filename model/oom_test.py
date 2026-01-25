# Test script
import torch
from model import GPT, GPTConfig

config = GPTConfig(
    vocab_size=256,
    n_layer=24,
    n_head=8,
    n_embd=512,
    num_streams=4,
    block_size = 1024,
    swiglu_hidden_factor=None,
    use_swiglu=True,
)

model = GPT(config).cuda()
x = torch.randint(0, 256, (5, 512)).cuda()

print(f"Memory before forward: {torch.cuda.memory_allocated()/1e9:.2f}GB")
logits, loss = model(x, x)
loss.backward()
print(f"Memory after backward: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")