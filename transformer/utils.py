import torch

def generate_casual_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
