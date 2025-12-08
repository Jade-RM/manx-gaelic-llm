import re
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gradio as gr

# Plug corpus into llm
# Use the uploaded file directly
with open("manx_corpus.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]

print(f"Loaded corpus with {len(corpus)} lines.")

# Define special tokens early
start_token = "[SOS]"
stop_token = "[EOS]"
user_token = "[USER]"
bot_token = "[BOT]"

# Initial vocabulary (chars + </w>)
unique_chars = set()
for doc in corpus:
    for char in doc:
        unique_chars.add(char)

vocab = sorted(list(unique_chars))
end_of_word = "</w>"
vocab.append(end_of_word)

# Add special conversational tokens to the initial vocab so BPE doesn't try to break them up 
if user_token not in vocab:
    vocab.append(user_token)
if bot_token not in vocab:
    vocab.append(bot_token)

print("\nInitial vocabulary:")
print(vocab)
print(f"Vocabulary size: {len(vocab)}")

# Word splits - Modified to handle special tokens as atomic units
word_splits = {}
for doc in corpus:
    words = doc.split(" ")
    for word in words:
        if not word:
            continue
        # If it's a special token, treat it as a single unit without splitting characters or adding </w>
        if word == user_token or word == bot_token:
            word_tuple = (word,)
        else:
            # For regular words, split into characters and append </w>
            char_list = list(word) + [end_of_word]
            word_tuple = tuple(char_list)
        word_splits[word_tuple] = word_splits.get(word_tuple, 0) + 1

print("\nPre-tokenized word frequencies:")
# Limit printing for large corpora
print(list(word_splits.items())[:20])
print("...")


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

num_merges = 250
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
# Limit printing for large vocabularies
print(final_vocab_sorted[:50])
print("...")


# Build lookup tables
# Modified: Include start and stop tokens in vocabulary
final_vocab_list = sorted(list(set(vocab)))

# Ensure all special tokens are in the final vocabulary list (tokens moved to the start of program)
if start_token not in final_vocab_list:
    final_vocab_list.append(start_token)
if stop_token not in final_vocab_list:
    final_vocab_list.append(stop_token)
if user_token not in final_vocab_list:
    final_vocab_list.append(user_token)
if bot_token not in final_vocab_list:
    final_vocab_list.append(bot_token)

stoi = {s: i for i, s in enumerate(final_vocab_list)}
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

print(f"\nFinal vocabulary size (including start and stop tokens): {vocab_size}")
print(f"Start token ID: {stoi[start_token]}")
print(f"Stop token ID: {stoi[stop_token]}")
print(f"User token ID: {stoi[user_token]}")
print(f"Bot token ID: {stoi[bot_token]}")


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
            # Initialize new_symbols here before breaking
            new_symbols = symbols
            break
        # Upgrade this BPE later find the highest priority merge.
        # Taking the first merge is fine at this stage.
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
    # Ensure user_token and bot_token are defined within this scope for direct use
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

# Transformer
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
        self.dropout = nn.Dropout(0.1) # Added dropout

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # Applied dropout here
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd): # Corrected: Add n_embd to init signature
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
# Modified: Get id for all special tokens and append/prepend to each line in the corpus
sos_token_id = stoi[start_token]
eos_token_id = stoi[stop_token]
# Get ids for the new conversational tokens
user_token_id = stoi[user_token]
bot_token_id = stoi[bot_token]

for line in corpus:
    encoded_line = encode(line)
    # Prepend start token and append stop token
    data.extend([sos_token_id] + encoded_line + [eos_token_id])

data = torch.tensor(data, dtype=torch.long)

def get_batch(batch_size=16, block_size=64):
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

model = TinyTransformer()
# Adjusted learning rate
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

print("\n--- Starting model training ---")
# Increased training steps for larger corpus
num_training_steps = 30000
for step in range(num_training_steps):
    xb, yb = get_batch(batch_size=16, block_size=block_size)
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0: # Print loss less frequently for more steps
        print(f"Step {step}, Loss: {loss.item():.4f}")
print("---- Model training complete ---")

# Text generation
def generate(model, start, max_new_tokens=30, temperature=0.8, top_k=None, top_p=None):
    model.eval()
    sos_token_id = stoi[start_token]
    eos_token_id = stoi[stop_token]
    user_token_id = stoi[user_token]
    bot_token_id = stoi[bot_token]

    start_ids = [sos_token_id] + encode(start)
    idx = torch.tensor([start_ids], dtype=torch.long)

    # Keep track of the length of the prompt to extract only newly generated tokens in the displayed response
    prompt_length = idx.shape[1]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Apply temperature
        logits = logits / temperature

        # Apply Top-K sampling if top_k is not None and top_k > 0
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Apply Top-P (Nucleus) sampling if top_p is not None and top_p < 1.0
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Check if the sampled token is a stop token (EOS, USER, or BOT)
        if next_id.item() == eos_token_id or next_id.item() == user_token_id or next_id.item() == bot_token_id:
            break # Stop generation if any of these special tokens are predicted

        idx = torch.cat((idx, next_id), dim=1)

    # Decode *only* the newly generated tokens after the prompt
    newly_generated_indices = idx[0, prompt_length:].tolist()

    # Filter out special tokens from the newly generated indices during decoding so that they are not displayed
    # This specifically removes SOS, EOS, USER, BOT from the final decoded string,
    # as the model might generate them before breaking, or they might be part of the prompt.
    filtered_generated_indices = [i for i in newly_generated_indices if i != sos_token_id and i != eos_token_id and i != user_token_id and i != bot_token_id]

    return decode(filtered_generated_indices)

# Build and continue a conversation from prompts and chat history
def build_prompt_from_history(history):
    parts = []
    for msg in history:
        if msg["role"] == "user":
            parts.append(f"{user_token} {msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"{bot_token} {msg['content']}")
    parts.append(bot_token) # Add the bot_token to indicate it's the chatbot's turn to respond
    return " ".join(parts)

# Generate a response based on the prompt and chat history
def generate_response(history, user_message_content, temperature, max_tokens, top_k, top_p):
    history = history or []
    user_message_content = user_message_content.strip()
    if not user_message_content:
        return history, None

    # Add the current user message to the history
    history.append({"role": "user", "content": user_message_content})

    # Build the chatbot's prompt from the updated history
    prompt = build_prompt_from_history(history)

    top_k_val = int(top_k) if top_k is not None and top_k > 0 else None
    top_p_val = float(top_p) if top_p is not None else None

    # Generate the response, returning only the new part and not displaying the history
    response_text = generate(model, prompt, max_new_tokens=int(max_tokens),
                          temperature=float(temperature), top_k=top_k_val, top_p=top_p_val)
    response_text = response_text.strip() # Ensure no leading/trailing whitespace

    # Append only the bot's new response to the history
    history.append({"role": "assistant", "content": response_text})

    return history, None

# Gradio interface, conversational chatbot with chat bubbles
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
