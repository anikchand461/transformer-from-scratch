from layers import *
import torch 
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=5000):
        super().__init__()

        # input block (embedding + positional encoding)
        self.input_block = InputBlock(vocab_size, d_model, max_len)

        # attention layers 
        self.mha_layers = nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        )

        # feed forward layer 
        self.ffn_layers = nn.ModuleList(
            [FeedForward(d_model) for _ in range(num_layers)]
        )

        # layerNorms 
        self.norm1_layers = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        self.norm2_layers = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

    def forward(self, X):
        # X : (batch, seq_len)

        X = self.input_block(X)

        for i in range(len(self.mha_layers)):

            # multi head attention 
            attn_out = self.mha_layers[i](X)
            X = self.norm1_layers[i](X + attn_out)

            # feed forward 
            ffnn_out = self.ffn_layers[i](X)
            X = self.norm2_layers[i](X + ffnn_out)

        return X



