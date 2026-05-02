from layers import *
import torch 
import torch.nn import nn

class Encoder(nn.Module):
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=5000):
        super().__init__()

        # input block (embedding + positional encoding)
        self.input_block = InputBlock(vocab_size, d_model, max_len)

        # attention layers 
        self.mha_layers = nn.ModuleList()
