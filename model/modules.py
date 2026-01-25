import torch
import torch.nn as nn
import math

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("Flash Attention v2 available - will use optimized kernels")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Flash Attention v2 not found - falling back to PyTorch native (still uses Flash if available)")


class RoPEPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for rotary embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len != self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len

            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))  # (T, dim/2)

            self._cos_cached = freqs.cos()  # (T, dim/2)
            self._sin_cached = freqs.sin()  # (T, dim/2)

    
    def forward(self, x, seq_len=None):
        """
        Apply rotary position embeddings to input tensor.

            x: Input tensor of shape (batch, seq_len, n_heads, head_dim)
            seq_len: Sequence length (if None, inferred from x)
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return apply_rotary_pos_emb(x, self._cos_cached, self._sin_cached)


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary position embeddings to input tensor.
    
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim)
        cos: Cosine values of shape (seq_len, head_dim)
        sin: Sine values of shape (seq_len, head_dim)
    """
    assert x.shape[-1] == cos.shape[-1] * 2

    # Split x into two halves along the head dimension
    x1, x2 = x.chunk(2, dim=-1)
    
    # Apply rotation
    # Reshape cos/sin to broadcast correctly
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Rotary embedding formula
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional Flash Attention v2.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # RoPE for positional embeddings
        self.rope = RoPEPositionalEmbedding(
            dim=config.n_embd // config.n_head,  # head_dim
            max_seq_len=config.block_size * 2,  # Allow some extrapolation
        )
        
        # Flash Attention flag
        self.use_flash_attn = FLASH_ATTN_AVAILABLE
    
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (B, T, n_head, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)
        
        # Apply RoPE to queries and keys
        q = self.rope(q, seq_len=T)
        k = self.rope(k, seq_len=T)
        
        if self.use_flash_attn and self.training:
            # Flash Attention v2 expects (B, T, n_head, head_dim)
            # It handles causal masking internally
            y = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                softmax_scale=1.0 / math.sqrt(head_dim)
            )
        else:
            # Fallback to scaled_dot_product_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            
            y = y.transpose(1, 2)
        
        # Re-assemble all head outputs side by side
        y = y.contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    """
    def __init__(self, config, hidden_factor=None):
        super().__init__()
        
        if hidden_factor is None:
            # Use 8/3 ratio like LLaMA
            hidden_dim = int(2 * config.n_embd * 2 / 3)
            # Round to nearest multiple of 256 for efficiency
            hidden_dim = 256 * ((hidden_dim + 255) // 256)
        else:
            hidden_dim = int(config.n_embd * hidden_factor)
        
        # Gate projection (for Swish activation)
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        # Up projection (for element-wise multiplication)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        # Down projection
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # SwiGLU: Swish(x @ W_gate) ⊙ (x @ W_up)
        gate = torch.nn.functional.silu(self.w_gate(x))  # Swish/SiLU activation
        up = self.w_up(x)
        x = gate * up  # Element-wise multiplication
        x = self.w_down(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """
    Standard MLP with GELU
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, ndim, bias=False, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
    
    def forward(self, x):
        # Compute RMS
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # Normalize
        x_normed = x / rms
        
        # Apply weight and bias
        output = x_normed * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output