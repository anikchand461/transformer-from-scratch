from transformer import *
import torch 

tfmr = Transformer(
    vocab_size= 10000,
    d_model=4,
    num_heads=2,
    num_layers=6
)

X = torch.randint(0, 10000, (2, 5))
y = torch.randint(0, 10000, (2, 5))

out = tfmr(X, y)

print(out)
print(out.shape)
