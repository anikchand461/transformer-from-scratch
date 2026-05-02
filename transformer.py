import torch 
import torch.nn as nn

from encoder import Encoder

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()

        # encoder 
        self.encoder = Encoder(
            vocab_size = vocab_size,
            d_model = d_model,
            num_heads = num_heads,
            num_layers = num_layers
        )

        # output projection 
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # src : (batch, seq_len)
        enc_out = self.encoder(src)

        # logits 
        logits = self.fc_out(enc_out)

        return logits
