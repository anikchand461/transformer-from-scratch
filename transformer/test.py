import torch
import torch.nn as nn
from transformer import Transformer


# ---------------- DATA ----------------
data = [
    "i love machine learning",
    "i love deep learning",
    "i love transformers",
    "i like neural networks",
    "you love machine learning",
    "you are learning transformers",
    "we are building models",
    "they are training neural networks",
    "he is studying artificial intelligence",
    "she is learning data science",
    "machine learning is powerful",
    "deep learning is very powerful",
    "transformers are very powerful",
    "neural networks learn patterns",
    "models learn from data",
    "data science is interesting",
    "artificial intelligence is the future",
    "learning ai is fun",
    "we are working on models",
    "we are building transformers",
    "we are learning attention mechanisms",
    "attention is very useful",
    "attention helps models learn context",
    "context is important in language",
    "language models predict next words",
    "this model predicts text",
    "this is a simple dataset",
    "we are training a transformer",
    "this transformer learns patterns",
    "patterns are important in data",
] * 10  # repeat to enlarge dataset

# ---------------- VOCAB ----------------
def build_vocab(data):
    words = []
    for sentence in data:
        words.extend(sentence.lower().split())

    vocab = {word: i + 4 for i, word in enumerate(set(words))}

    vocab["<PAD>"] = 0
    vocab["<SOS>"] = 1
    vocab["<EOS>"] = 2
    vocab["<UNK>"] = 3

    return vocab


# ---------------- ENCODE ----------------
def encode(sentence, vocab, max_len=12):
    tokens = ["<SOS>"] + sentence.lower().split() + ["<EOS>"]

    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]

    # pad / truncate
    ids = ids[:max_len] + [vocab["<PAD>"]] * (max_len - len(ids))

    return ids


# ---------------- PREP ----------------
vocab = build_vocab(data)
vocab_size = len(vocab)

encoded = [encode(s, vocab) for s in data]
dataset = torch.tensor(encoded)

src = dataset
tgt_input = dataset[:, :-1]
tgt_output = dataset[:, 1:]


# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- MODEL ----------------
model = Transformer(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=16,
    num_layers=4
).to(device)

src = src.to(device)
tgt_input = tgt_input.to(device)
tgt_output = tgt_output.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)


# ---------------- TRAIN ----------------
epochs = 800

for epoch in range(epochs):

    model.train()
    optimizer.zero_grad()

    output = model(src, tgt_input)
    # (batch, seq_len, vocab)

    batch_size, seq_len_out, vocab_size = output.shape
    seq_len_tgt = tgt_output.shape[1]

# align output to target length
    output = output[:, :seq_len_tgt, :]

    output = output.reshape(batch_size * seq_len_tgt, vocab_size)
    tgt = tgt_output.reshape(batch_size * seq_len_tgt)

    loss = criterion(output, tgt)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# ---------------- GENERATE ----------------
def generate(model, src, vocab, max_len=12):

    model.eval()
    device = src.device

    inv_vocab = {v: k for k, v in vocab.items()}

    tokens = [vocab["<SOS>"]]

    for _ in range(max_len):

        tgt = torch.tensor([tokens], device=device)

        with torch.no_grad():
            out = model(src, tgt)

            probs = torch.softmax(out[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_token)

        if next_token == vocab["<EOS>"]:
            break

    return [inv_vocab[t] for t in tokens]


# ---------------- TEST ----------------
test_sentences = [
    "hello how are",
    "machine learning is",
    "i am",
    "transformers are",
    "attention mechanism is"
]

for sentence in test_sentences:
    src_test = torch.tensor([encode(sentence, vocab)]).to(device)
    output = generate(model, src_test, vocab)
    print(f"\nInput: {sentence}")
    print("Generated:", output)
