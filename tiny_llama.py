import re
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Plug corpus into llama
with open("manx_gaelic_corpus.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]

print("Training Corpus:")
for doc in corpus:
    print(doc)

# Initial vocabulary (chars + </w>)
unique_chars = set()
for doc in corpus:
    for char in doc:
        unique_chars.add(char)

vocab = sorted(list(unique_chars))
end_of_word = "</w>"
vocab.append(end_of_word)

print("\nInitial vocabulary:")
print(vocab)
print(f"Vocabulary size: {len(vocab)}")

# Word splits
word_splits = {}
for doc in corpus:
    words = doc.split(" ")
    for word in words:
        if word:
            char_list = list(word) + [end_of_word]
            word_tuple = tuple(char_list)
            word_splits[word_tuple] = word_splits.get(word_tuple, 0) + 1

print("\nPre-tokenized word frequencies:")
print(word_splits)

# BPE training
def get_pair_stats(splits):
    pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq
    return pair_counts

def merge_pair(pair_to_merge, splits):
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[tuple(new_symbols)] = freq
    return new_splits

num_merges = 50
merges = {}
current_splits = word_splits.copy()

print("\n--- Starting BPE Merges ---")
for i in range(num_merges):
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        break
    best_pair = max(pair_stats, key=pair_stats.get)
    current_splits = merge_pair(best_pair, current_splits)
    new_token = best_pair[0] + best_pair[1]
    vocab.append(new_token)
    merges[best_pair] = new_token

print("\n--- BPE merges complete ---")
print(f"Final vocabulary size: {len(vocab)}")
final_vocab_sorted = sorted(list(set(vocab)))
print("\nFinal vocabulary (sorted):")
print(final_vocab_sorted)

# Build lookup tables
stoi = {s: i for i, s in enumerate(final_vocab_sorted)}
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

# Apply merges
def bpe_encode_word(word):
    symbols = list(word) + [end_of_word]
    while True:
        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
        merge_candidates = [(p, merges[p]) for p in pairs if p in merges]
        if not merge_candidates:
            break
        best_pair, merged_token = merge_candidates[0]
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == best_pair:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols

def encode(text):
    ids = []
    for word in text.split(" "):
        for token in bpe_encode_word(word):
            if token in stoi:
                ids.append(stoi[token])
    return ids

def decode(indices):
    tokens = [itos[i] for i in indices]
    text = "".join([t.replace(end_of_word, " ") for t in tokens])
    return text.strip()

# Transformer
block_size = 16
n_embd = 64
n_head = 4
n_layer = 2

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# Training
data = []
for line in corpus:
    data.extend(encode(line))
data = torch.tensor(data, dtype=torch.long)

def get_batch(batch_size=4, block_size=8):
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

model = TinyTransformer()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

print("Starting training...")
for step in range(200):
    xb, yb = get_batch(batch_size=16, block_size=block_size)
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Text generation
def generate(model, start, max_new_tokens=30):
    model.eval()
    idx = torch.tensor([encode(start)], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return decode(idx[0].tolist())

print("\n--- SAMPLE GENERATION ---")
print(generate(model, "Ta mee"))

