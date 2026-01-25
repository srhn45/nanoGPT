import torch
from hyperconnections import HyperConnections, get_init_and_expand_reduce_stream_functions
from modules import CausalSelfAttention, MLP, RMSNorm
from torch import nn

class HyperBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, bias=config.bias)
        
        # Get hyper connection initialization function
        init_hyper_conn_fn, _, _ = get_init_and_expand_reduce_stream_functions(
            num_streams=config.num_streams,
            dim=config.n_embd,
            add_stream_embed=False
        )
        
        # Create attention with hyper connections
        self.attn = init_hyper_conn_fn(
            branch=CausalSelfAttention(config),
            tanh=True,
            channel_first=False,
            dropout=config.dropout,
            add_branch_out_to_residual=True
        )
        
        self.ln_2 = RMSNorm(config.n_embd, bias=config.bias)
        
        # Create MLP with hyper connections
        self.mlp = init_hyper_conn_fn(
            branch=MLP(config),
            tanh=True,
            channel_first=False,
            dropout=config.dropout,
            add_branch_out_to_residual=True
        )

    def forward(self, x):
        # input already fanned out
        # Attention path
        x = x + self.attn(self.ln_1(x))
        # MLP path
        x = x + self.mlp(self.ln_2(x))
        return x