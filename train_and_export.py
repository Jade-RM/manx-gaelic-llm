import torch
import torch.optim as optim

from tokenizer import (
    corpus,
    encode,
    stoi,
    start_token,
    stop_token,
    save_tokenizer
)

from model import TinyTransformer, block_size

# ----------------------------
# Build training data
# ----------------------------
data = []
sos_token_id = stoi[start_token]
eos_token_id = stoi[stop_token]

for line in corpus:
    data.extend([sos_token_id] + encode(line) + [eos_token_id])

data = torch.tensor(data, dtype=torch.long)

def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ----------------------------
# Model + optimizer
# ----------------------------
model = TinyTransformer(vocab_size=len(stoi))
optimizer = optim.AdamW(model.parameters(), lr=2e-4)

# ----------------------------
# Training loop
# ----------------------------
print("🚀 Training started")
num_steps = 36000

for step in range(num_steps):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step {step:05d} | Loss {loss.item():.4f}")

# ----------------------------
# Save everything
# ----------------------------
torch.save(model.state_dict(), "manx_model.pt")
save_tokenizer("tokenizer.pt")

print("Training complete")
print("Exported: manx_model.pt + tokenizer.pt")
