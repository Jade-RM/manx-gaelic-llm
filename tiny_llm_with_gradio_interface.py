import re
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gradio as gr

start_token = "[SOS]"
stop_token = "[EOS]"
user_token = "[USER]"
bot_token = "[BOT]"

vocab = [] # populated from loaded tokenizer
end_of_word = "</w>"

merges = {}

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

# Apply merges
def bpe_encode_word(word):
    # If the word itself is a special token, return it as a single unit
    if word == user_token or word == bot_token:
        return [word]

    symbols = list(word) + [end_of_word]
    while True:
        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
        merge_candidates = [(p, merges[p]) for p in pairs if p in merges]
        if not merge_candidates:
            new_symbols = symbols
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
    return new_symbols

def encode(text):
    ids = []
    user_token_id_local = stoi[user_token]
    bot_token_id_local = stoi[bot_token]

    for word in text.split(" "):
        if word == user_token:
            ids.append(user_token_id_local)
        elif word == bot_token:
            ids.append(bot_token_id_local)
        else:
            for token in bpe_encode_word(word):
                if token in stoi:
                    ids.append(stoi[token])
    return ids

def decode(indices):
    tokens = [itos[i] for i in indices]
    text = "".join([t.replace(end_of_word, " ") for t in tokens])
    return text.strip()

def save_tokenizer(path="tokenizer.pt"):
    torch.save({
        "stoi": stoi,
        "itos": itos,
        "merges": merges,
        "vocab_size": vocab_size
    }, path)

def load_tokenizer(path="tokenizer.pt"):
    data = torch.load(path, map_location="cpu")
    return data

block_size = 64
n_embd = 128
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(0.1),
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
    def __init__(self, vocab_size):
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

tok_data = load_tokenizer("tokenizer.pt")
stoi = tok_data["stoi"]
itos = tok_data["itos"]
merges = tok_data["merges"] # Assign merges here as it's used by bpe_encode_word
vocab_size = tok_data["vocab_size"]

model = TinyTransformer(vocab_size)
model.load_state_dict(torch.load("manx_model.pt", map_location="cpu"))
model.eval()

def generate(model, start, max_new_tokens=30, temperature=0.8, top_k=None, top_p=None):
    model.eval()
    sos_token_id = stoi[start_token]
    eos_token_id = stoi[stop_token]
    user_token_id = stoi[user_token]
    bot_token_id = stoi[bot_token]

    start_ids = [sos_token_id] + encode(start)
    idx = torch.tensor([start_ids], dtype=torch.long)

    prompt_length = idx.shape[1]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eos_token_id or next_id.item() == user_token_id or next_id.item() == bot_token_id:
            break

        idx = torch.cat((idx, next_id), dim=1)

    newly_generated_indices = idx[0, prompt_length:].tolist()

    filtered_generated_indices = [i for i in newly_generated_indices if i != sos_token_id and i != eos_token_id and i != user_token_id and i != bot_token_id]

    return decode(filtered_generated_indices)

def build_prompt_from_history(history):
    parts = []
    for msg in history:
        if msg["role"] == "user":
            parts.append(f"{user_token} {msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"{bot_token} {msg['content']}")
    parts.append(bot_token)
    return " ".join(parts)

def generate_response(history, user_message_content, temperature, max_tokens, top_k, top_p):
    history = history or []
    user_message_content = user_message_content.strip()
    if not user_message_content:
        return history, None

    history.append({"role": "user", "content": user_message_content})

    prompt = build_prompt_from_history(history)

    top_k_val = int(top_k) if top_k is not None and top_k > 0 else None
    top_p_val = float(top_p) if top_p is not None else None

    response_text = generate(model, prompt, max_new_tokens=int(max_tokens),
                          temperature=float(temperature), top_k=top_k_val, top_p=top_p_val)
    response_text = response_text.strip()

    history.append({"role": "assistant", "content": response_text})

    return history, None

with gr.Blocks() as demo:
    gr.Markdown("# Tiny Manx Chatbot")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(elem_id="chatbox", height=420, type='messages')
            user_input = gr.Textbox(label="Your message", placeholder="Say something to the Manx chatbot...")
            with gr.Row():
                send_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear chat")
        with gr.Column(scale=1):
            max_tokens = gr.Slider(10, 256, value=30, step=1, label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
            top_k = gr.Slider(0, vocab_size, value=0, step=1, label="Top-K (0 to disable)")
            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-P (nucleus)")

    def submit_fn(history, user_message_content, temperature, max_tokens, top_k, top_p):
        return generate_response(history, user_message_content, temperature, max_tokens, top_k, top_p)

    user_input.submit(submit_fn, inputs=[chatbot, user_input, temperature, max_tokens, top_k, top_p], outputs=[chatbot, user_input])
    send_btn.click(submit_fn, inputs=[chatbot, user_input, temperature, max_tokens, top_k, top_p], outputs=[chatbot, user_input])
    clear_btn.click(lambda: [], None, chatbot)

print("\n--- Launching Gradio chat interface ---")

demo.launch(share=False)
