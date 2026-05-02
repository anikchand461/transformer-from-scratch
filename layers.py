import torch
import torch.nn as nn
import math as m

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = m.sqrt(d_model)

    def forward(self, X):
        return self.embedding(X) * self.scale

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # create positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        # position indices (0, 1, 2, ...)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # compute denominator term 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-m.log(10000.0) / d_model))

        # apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension
        pe = pe.unsqueeze(0)

        # register as buffer (not trainable)
        self.register_buffer("pe", pe)

    def forward(self, X):
        # X : (batch_size, seq_len, d_model)
        seq_len = X.size(1)

        # add positional encoding 
        return X + self.pe[:, :seq_len]

class SelfAttention(nn.Module):
    
    def __init__(self, d_model):
        super().__init__()

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

    def forward(self, X):

        # X : (batch, seq_len, d_model)
        Q = self.wq(X)  # (batch, seq_len, d_model)  ->  X @ wq + bias 
        K = self.wk(X)  # (batch, seq_len, d_model)
        V = self.wv(X)  # (batch, seq_len, d_model)

        # attention scores 
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)

        # scale 
        scores = scores / m.sqrt(X.size(-1))

        # softmax
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # weighted sum 
        output = torch.matmul(weights, V)  # (batch, seq_len, d_model) 

        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0 # this is like an if else .... if this condition will fail immediately stop the whole process 

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # same as the hingle head attention 
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # final projection 
        self.fc_out = nn.Linear(d_model, d_model)

        # layer normalization 
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        # 1. linear projections
        Q = self.wq(X)  # (batch, seq_len, d_model)
        K = self.wk(X)  # (batch, seq_len, d_model)
        V = self.wv(X)  # (batch, seq_len, d_model)
        
        # split into heads 
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)   # (batch, seq_len, num_heads, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 3. rearrange for attention  : swap dimesion 1 and 2 
        Q = Q.transpose(1, 2)   # (batch, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4. Atention score (same as before)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, head_dim)
        # scale 
        scores = scores / m.sqrt(self.head_dim)
        # softmax
        weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, head_dim)
        # weighted sum 
        output = torch.matmul(weights, V)  # (batch, num_heads, seq_len, head_dim)

        # combine heads
        output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)

        output = output.contiguous().view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)  d_model = num_heads * head_dim

        output = self.fc_out(output)   # this is doing just ... output @ wt + bias   -> a fully connected layer   # (batch, seq_len, d_model)

        output = output + X  # residual connection 
        output = self.norm(output)  # layer normalization 

        return output

class FeedForward(nn.Module):
    
    def __init__(self, d_model):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, X):
        return self.network(X)

class InputBlock(nn.Module):

    def __init__(self, vocab_size, d_model, max_len=5000):
        super().__init__()
        
        self.network = nn.Sequential(
            TokenEmbedding(vocab_size, d_model),
            PositionalEncoding(d_model, max_len),
        )

    def forward(self, X):
        # X : (batch, seq_len)
        return self.network(X)

# vocab_size = 10000
# d_model = 4
#
# embedding = TokenEmbedding(vocab_size, d_model)
#
# x = torch.randint(0, vocab_size, (2, 5))
#
# print(x.shape)
#
# out = embedding(x)
#
# print(out)
# print(out.shape)
#
# pe = PositionalEncoding(d_model)
# pev = pe(out)
#
# print(pev)
# print(pev.shape)

# attn = SelfAttention(d_model)
# output_se = attn(pev)
# print(output_se)
# print(output_se.shape)

# nh_attn = MultiHeadAttention(d_model, 2)
# output_mha = nh_attn(pev)
# print(output_mha)
# print(output_mha.shape)
#
# ffnn = FeedForward(d_model)
# output_final = ffnn(output_mha)
# print(output_final)
# print(output_final.shape)
#


vocab_size = 10000
d_model = 4

ib = InputBlock(vocab_size, d_model)
X = torch.randint(0, vocab_size, (2, 5))
out = ib(X)
print(out)
print(out.shape)










