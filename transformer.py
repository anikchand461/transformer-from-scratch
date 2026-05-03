import torch 
import torch.nn as nn
from utils import generate_casual_mask
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=5000):
        super().__init__()

        # encoder 
        self.encoder = Encoder(
            vocab_size = vocab_size,
            d_model = d_model,
            num_heads = num_heads,
            num_layers = num_layers,
            max_len = max_len
        )

        self.decoder = Decoder(
            vocab_size = vocab_size,
            d_model = d_model,
            num_heads = num_heads,
            num_layers = num_layers,
            max_len = max_len
        )

        # output projection 
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # src : (batch, src_len)
        # tgt : (batch, tgt_len)

        # encoder 
        enc_out = self.encoder(src)  # (batch, src_len, d_model)
        
        # create a mask for decoder 
        tgt_len = tgt.size(1)
        mask = generate_casual_mask(tgt_len).to(tgt.device)  # (1, 1, tgt_len, tgt_len)

        # decoder 
        dec_out = self.decoder(tgt, enc_out, mask)  # (batch, tgt_len, d_model)
                
        # logits : final projection 
        logits = self.fc_out(enc_out)   # (batch, tgt_len, vocab_size)

        return logits
