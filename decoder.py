from layers import *
import torch 
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=5000):
        super().__init__()

        # input : (embedding + pe)
        self.input_block = InputBlock(vocab_size, d_model, max_len)

        # masked self attention layers
        self.masked_mha_layers = nn.ModuleList(
            [MaskedMultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        )

        # cross attention layers 
        self.cross_attn_layers = nn.ModuleList(
            [CrossAttention(d_model, num_heads) for _ in range(num_layers)]
        )

        # feed forward network layer 
        self.ffn_layers = nn.ModuleList(
            [FeedForward(d_model) for _ in range(num_layers)]
        )

        # Layer Norm s 
        self.norm1_layers = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.norm2_layers = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.norm3_layers = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

    def forward(self, tgt, enc_out, mask):
        # tgt : (batch, tgt_len)
        # enc_out : (batch, src_len, d_model)
        # mask : (batch, 1, tgt_len, tgt_len)
        
        X = self.input_block(tgt)

        for i in range(len(self.masked_mha_layers)):

            # masked multi head attention 
            attn_out = self.masked_mha_layers[i](X, mask)
            X = self.norm1_layers[i](X+attn_out)

            # cross attention 
            ca_out = self.cross_attn_layers[i](X, enc_out, enc_out)
            X = self.norm2_layers[i](X + ca_out)

            # ffnn
            ffn_out = self.ffn_layers[i](X)
            X = self.norm3_layers[i](X + ffn_out)

        return X



