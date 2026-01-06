# tokenizer.py
import re
import collections
import torch

# Plug corpus into llm
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

# Add conversational tokens to the initial vocab 
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

num_merges = 350
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

final_vocab_list = sorted(list(set(vocab)))

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
    
# functionality check    
if __name__ == "__main__":
    print("Tokenizer OK")
    print("Vocab size:", vocab_size)
    print("Test encode/decode:", decode(encode("hello")))

