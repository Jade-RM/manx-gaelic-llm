# Step 1: Prepare Training Data

# Sample training data
corpus = [
 "Fastyr mie, kys t'ou?",
 "Fastyr mie, ta mee braew. Gura mie ayd. Kys t'ou hene?", 
 "Moghrey mie, kys t'ou?",
 "Moghrey mie, ta mee braew, gura mie ayd. As oo hene?",
 "Ta mee braew, gura mie ayd. As oo hene?",
 "Ta mee goll as gaccan. Kys t’ou, whooinney?",
 "Ta mee braew, gura mie ayd. Kys t'ou hene?",
 "Ta mee skee.",
 "Ta mee red beg skee.", 
 "Ta mee red beg çhing.",
 "Ta mee goll as gaccan, ghooinney",
 "Ta mee red beg skee, ghooinney",
 "Ta mee skee, ghooiney",
 "Cha nel mee feer vie. Ta mee çhing.",
 "Cha nel oo feer vie. T'ou çhing.",
 "Cha nel eh feer vie. T'eh çhing.",
 "Cha nel ee feer vie. T'ee çhing.",
 "Ta mee maynrey",
 "T'ou maynrey",
 "T'eh maynrey",
 "Smie lhiam shappal",
 "Cha mie lhiam shappal",
 "Smie lhiam shooyl",
 "Cha mie lhiam shooyl",
 "Smie lhiam roie",
 "Cha mie lhiam roie",
 "Smie lhiam snaue",
 "Cha mie lhiam snaue",
 "Smie lhiat shappal",
 "Cha mie lhiat shappal",
 "Kys t'an emshyr jiu?",
 "T'an emshyr braew jiu",
 "T'eh feayr",
 "T'eh fliugh",
 "T'eh geayagh",
 "T'eh çheh",
 "T'eh çhirrym",
 "T'eh ceau fliaghey",
 "T'eh ceau sniaghtey",
 "T'eh ceau sniaghtey garro",
 "Cred t'ou jannoo dagh laa?",
 "Cre gollrish yn emshyr jiu?",
 "T'eh fliugh ayns Mannin",
 "T'eh ceau fliaghey ayns Sostyn",
 "Vel oo beaghey ayns Doolish?",
 "Cha nel mee beaghey ayns Doolish. Ta mee beaghey ayns Purt le Moirrey",
 "Ta mee cummal ayns Doolish.",
 "Cha nel mee cummals ayns Doolish",
 "Ta mee mie dy liooar.",
 "Cha nel mee mie dy liooar.",
 "Ta mee feer çhing as feer skee.",
 "Moghrey mie!",
 "Fastyr mie!",
 "Oie vie!",
 "Oie vie, kys t'ou?"
 "C'raad voish t'ou?",
 "Ta mee voish Mannin.",
 "Ta mee voish Sostyn.",
 "Ta mee voish Nalbin.",
 "Ta mee voish Nerin.",
 "Ta mee voish Bretyn.",
 "T'eh voish Mannin.",
 "T'ee voish Sostyn.",
 "Ta shin voish Nalbin.",
 "Ta shiu voish Nerin.",
 "T'ad voish Bretyn.",
 "T'ad voish Mannin.",
 "Ta shin voish Sostyn.",
 "Vel Gaelg ayd?",
 "Ta Gaelg aym.",
 "Vel Baarle ayd?",
 "Ta Baarle as Gaelg ayd.",
 "C'red s'mie lhiat jannoo?",
 "S'mie lhiam roie.",
 "S'mie lhiam lhaih.",
 "S'mie lhiam gynsagh çhengaghyn.",
 "S'mie lhiam snaue.",
 "Ta mee roie.",
 "T'ou lhaih.",
 "T'eh lhaih.",
 "T'ee lhaih lioar.",
 "Ta shin gynsagh çhengaghyn.",
 "Ta shiu snaue.",
 "T'ad snaue.",
 "C’raad t’ou?",
 "Ta mee ayns shoh.",
 "T’ou ayns shen.",
 "Cha nel oo voish Mannin.",
 "Cha nel mee voish Mannin. Ta mee voish Sostyn.",
 "T’eh jesh.",
 "Cha nel ad cummal ayns Sostyn.",
 "Cha nel ad cummal ayns Nalbin.",
 "Vel shiu ayns yn thie?",
 "Ta mee gynsagh Gaelg son ta mee geearee gol dys Mannin.",
 "Ta mee gynsagh Baarle son ta mee geearee gol dys Sostyn.",
 "Ta mee gynsagh Yernish son ta mee geearee gol dys Nerin.",
 "T’eh beaghey ayns Italy as t’eh gee peetsey",
 "Ta shin laccal beaghey ayns Sostyn",
 "T'eh laccal beaghey ayns Doolish",
 "Vel eh giu caffee?",
 "Cha nel eh giu caffee, t'eh giu tey.",  
 "Vel ee gobbragh ayns Doolish?",  
 "Cha nel eh giu tey agh t’eh giu caffee.",
 "Cha nel ee laccal gobbragh", 
 "C’raad t’eh beaghey?",
 "C’raad t’ee gobbragh?",
 "T’eh feer skee son t’eh gobbragh ayns Doolish", 
 "T'ee feer skee son t'ee gobbragh jiu.", 
 "T’eh giu ushtey agh t’eh laccal giu caffee.",
 "Ta mee giu ushtey agh ta mee laccal giu feeyn jiarg.",
 "Ta shin laccal giu feeyn gial.",
 "Cha nel ee laccal gee peetsey.",  
 "Cha nel eh laccal gee curry." 
 "C’raad t’eh giu?", 
 "T’ee red beg corree.",
 "Ta caffee aym, ta mee giu caffee dagh laa.", 
 "Cha nel tey aym, son cha laik lhiam tey.",
 "Ta eeast aym as t’eh giu ushtey.",
 "Ta kayt aym agh cha nel eh gee feill.",
 "Ta conning aym agh cha nel eh gee carradjyn.",
 "Nagh vel oo beaghey ayns Doolish?", 
 "Nagh vel oo beaghey ayns Purt ny Hinshey.",
 "Nagh vel eh braew?",
 "Nagh laik lhiat gobbragh?",
 "Nagh vel moddey ayd?",
 "Nagh vel conning ayd?",
 "Nagh vel cabbyl ayd?",
 "Cre t’ou gee?",
 "Ta mee gee curry.",
 "Ta mee gee peetsey.",
 "Ta mee gee ooylyn as peearyn.",
 "Ta mee gee carradjyn.",
 "Cre t’eh giu?",
 "Ta mee giu tey",
 "Cre t’ee gynsagh?",
 "Ta mee gynsagh Gaelg ayns y voghrey.",
 "Ta mee gynsagh Yernish ayns y voghrey.",
 "Ta fys aym c’raad t’ou beaghey.",
 "Vel fys ayd cre t’ee gee?",
 "Cha nel moddey doo aym.",
 "Vel kayt bane ayd?",
 "Share lhiam gynsagh Gaelg",  
 "Nhare lhiat ushtey ny caffee?",
 "Share lesh beaghey ayns thie.",   
 "Share lhee gobbragh ayns Rhumsaa.",
 "S’laik lhiam ushtey agh share lhiam feeyn.", 
 "Cre s’laik lesh gee ayns y voghrey?", 
 "Cre share lhee giu ayns y voghrey?", 
 "Vel fys ayd cre s’laik lesh jannoo?",
 "T’ee gee eeast dagh laa son s’laik lhee eeast.",
 "T'eh gee feill dagh laa son s'laik lesh feill.",
 "Nhare lesh feeyn jiarg ny feeyn gial?",
 "Vel baaghyn ayd?",
 "Ta, ta moddey aym.",
 "Vel moddey ayd hene?"
 "Ta kayt aym.",
 "Vel kayt ayd hene?"
 "Ta conning aym.",
 "Vel conning ayd hene?",
 "Ta cabbyl aym.",
 "Ta doo-oalee aym.",
 "Ta kayt doo aym.",
 "Ta conning bane aym.",
 "T'eh conning doo aym.",
 "Ta moddey doo aym.",
 "Ta cabbyl dhone aym.",
 "Cre t’ou jannoo Jesarn?",
 "C’raad t’eh shappal Jesarn?",
 "S’laik lhiam cloie kiaull ayns y voghrey", 
 "Share lhee gynsagh Gaelg Jesarn",
 "Cre t’ou kionnagh?",
 "Quoi ta jannoo arran?", 
 "Ta daa-wheeyl eck as t’eh beg", 
 "Cha nel fys eck cre t’eh jannoo!",
 "Vel fys echey quoi ta beaghey ayns ta thie bane shoh?",
 "Cha nel kayt aym.",           
 "Ta kayt ayd.",
 "Cha nel moddey ayd.",
 "Ta thie echey.",   
 "Cha nel thie echey.",
 "Ta kayt eck.",
 "Ta mee red beg çhing jiu, son ren mee giu feeyn jea.",
 "Share lhiam tey, agh cha nel tey aym.",
 "Cha nel mee laccal giu feeyn jiarg reesht.",
 "Ren eh goll dys y thie-bee agh cha ren eh gee jinnair.",
 "Moghrey mie, vel oo mie dy liooar?",
 "Cha nel. Cha nel mee feer vie.",
 "Cha nel mee feer vie edyr!",
 "Vel oo caillit?",
 "Ta, ta mee caillit",
 "Ta mee toiggal",
 "Cre t’ou laccal giu?",
 "Cha laik lhiam espresso",
 "C’raad t’ou beaghey eisht?", 
 "Ta mee beaghey ayns tholtan ayns balley marrey.", 
 "S’treih shen.C’raad t’ou laccal beaghey?" 
 "Ta mee laccal beaghey ayns ta thie bane shoh.",
 "Ta mee toiggal.",
 "Ta fys aym.",
 "Cre t’ou jannoo ayns y voghrey? ",
 "Cre t’ou giu ayns y voghrey?",
 "Cre t’ou gee ayns y voghrey?",
 "Ta Gaelg ain",
 "Cha nel argid ain", 
 "Ta argid eu",    
 "Cha nel Gaelg eu",
 "Ta folt oc",   
 "Cha nel kayt oc",  
 "Ta daa-wheeyl ec Jo",  
 "Cha nel folt ec Jo",     
 "Ta kayt ec Juan",
 "Cha nel penn ec Juan",
 "Ren eh goll dys y thie-bee agh cha ren eh gee arran",
 "T’eh feer skee jiu son ren eh gobbragh jea.",
 "Cha laik lhee gobbragh Jesarn.",
 "Cre ren oo kionnagh ayns Doolish jea?",
 "C’raad ren eh gee peetsey Jedoonee?",
 "Quoi uss?",
 "Mish Juan, quoi uss?",
 "Mish Moirrey. Kys t’ou, Yuan?", 
 "Ta mee castreycair, Voirrey. Kys t’ou hene?",
 "Ta mee red beg çhing. Ren mee giu rour lhune jea.",
 "Cha nel mee toiggal.", 
 "T’ou toiggal, vel?",
 "Ta. Ta mee toiggal.",
 "Ta, ta mish giu rour dagh laa.",
 "S’laik lhiam giu, son ta mee maynrey tra ta mee giu.",
 "S'laik lhiam gee peetsey, son ta mee maynrey tra ta mee gee caashey.",
 "Vel fys ayd c’raad ta shapp?",
 "Ta mee laccal kionnagh arran.",
 "Ta mee laccal kionnagh medshin.",
 "Ta mee laccal kionnagh bainney.",
 "Gow my leshtal, cha nel fys aym.",
 "Mie dy liooar, slane lhiat.",
 "Slane lhiat, hee’m oo.",
 "Hee'm shiu.",
 "Ren mee gynsagh?",
 "Nagh ren mee goll?",
 "Ren oo gee?",
 "Nagh ren oo cloie?",
 "Ren eh giu?",
 "Nagh ren eh gee?",
 "Ren ee gobbragh?",
 "Ren mee shappal.",
 "Cha ren mee shappal."
 "Ren oo gee peetsey.",
 "Cha ren oo gee.",
 "Ren eh giu caffee.",
 "Cha ren eh giu.",
 "Ren ee gynsagh.",
 "Cha ren ee gobbragh.",
 "C’raad t’ou laccal beaghey?",
 "Moghrey mie, vel oo mie dy liooar?",
 "Cha nel, cha nel mee feer vie edyr! Cha nel fys aym c’raad ta mee.", 
 "Oh, vel oo caillit?",
 "Ta, ta mee caillit.",
 "Ta mee toiggal. T’ou ayns Purt ny Hinshey.",
 "Cha nel mee laccal ve ayns Purt ny Hinshey",
 "C’raad t’ou laccal ve?",
 "Ta mee laccal ve ayns thie-bee ayns Doolish.",
 "Ta thie-bee feer vie ayns Purt ny Hinshey, nagh vel fys ayd?",
 "Vel oo laccal goll?",
 "Ta, ta mee laccal goll ayns y thie-bee!",
 "Mie dy liooar.", 
 "Cre t’ou laccal giu?",
 "Ta mee laccal espresso doobyl my sailt.",
 "Cha laik lhiam espresso. Ta mish laccal soo-mess",
 "S’laik lhiam soo-mess ayns y voghrey.",
 "S’mie shen. Slaynt vie.",
 "Ta cooat jiarg aym.",
 "Ta cooat doo aym.",
 "Ta cooat bwee as jeenym gorrym aym.",
 "Ta cooat gorrym-jiarg as jeenym doo aym.",
 "Ta kayt lheeah as moddey bane aym.",
 "Ta edd geayney as cooat jiarg-bwee aym.", # more colour sentences
 "By vie lhiat jough?",
 "Cha by vie lhiam.",
 "By vie lhiam.",
 "C'red by vie lhiat giu?",
 "By vie lhiam ushtey.",
 "By vie lhiam bainney, my sailt.",
 "By vie lhiam feeyn.",
 "By vie lhiam pynt sharoo.",
 "By vie lhiam pynt lhune.",
 "By vie lhiam soo-mess lesh rio.",
 "By vie lhiat caffee?",
 "Gura mie ayd, whooinney.",
 "She dty vea.",
 "By vie lhiam caffee lesh bainney agh gyn shugyr",
 "By vie lhiam caffee gyn shugyr",
 "C'red by vie lhiat gee?",
 "By vie lhiam peetsey",
 "By vie lhiam peetsey lesh caashey, unnish as garleid",
 "Fastyr mie. C'red by vie lhiat?",
 "By vie lhiam boteil dy bainney as arran",
 "Vel argid dy-liooar ayd?",
 "Cha nel argid aym.",
 "Ta, ta mee argid dy-liooar aym."
]

print("Training Corpus:")
for doc in corpus:
    print(doc)

# Initialize vocabulary with unique characters
unique_chars = set()
for doc in corpus:
    for char in doc:
        unique_chars.add(char)
        
vocab = list(unique_chars)
vocab.sort() # For consistent order of characters, making the vocabulary list predictable

# Add a special end-of-word token
end_of_word = "</w>"
vocab.append(end_of_word)

print("Initial vocabulary:")
print(vocab)
print(f"Vocabulary size: {len(vocab)}")

# Pre-tokenize the corpus: Split into words and then characters
# We'll split by space for simplicity and add the end-of-word token
word_splits = {}
for doc in corpus:
    words = doc.split(' ')
    for word in words:
        if word:
            char_list = list(word) + [end_of_word]
            # Use tuple for immutibility if storing counts later
            word_tuple = tuple(char_list)
            if word_tuple not in word_splits:
                word_splits[word_tuple] = 0
            word_splits[word_tuple] += 1 # Count frequency of each initial word split
            
print("\nPre-tokenized word frequencies:")
print(word_splits)
 
import collections
 
def get_pair_stats(splits):
     # defaultdict doesn't raise an error if a key doesn't exist. 
     # It creates the key and assigns a default value.
     # int is the default factory called when a key is created.
     # int called with no arguments returns 0.
    pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq # Add the frequency of the word to the pair count
    return pair_counts 
     
 # This function takes a specific pair that we want to combine and the current splits.
def merge_pair(pair_to_merge, splits):
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        while i < len(symbols):
             # If the current symbol and next symbol match the pair to merge_pair
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                new_symbols.append(merged_token)
                i += 2 # Skip the next symbol
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[tuple(new_symbols)] = freq # Use the updated symbol list as key
    return new_splits
     
num_merges = 150
# Stores merge rules, e.g., {('a', 'b'): 'ab'}
merges = {}
current_splits = word_splits.copy() # Start with initial word splits

print("\n--- Starting BPE Merges ---")
print(f"Initial Splits: {current_splits}")
print("-" * 30)

for i in range(num_merges):
    print(f"\nMerge Iteration {i+1}/{num_merges}")
    
    # Calculate pair frequencies
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("No more pairs to merge.")
        break
    # Print top 5 pairs for inspection
    sorted_pairs = sorted(pair_stats.items(), key=lambda item: item[1], reverse=True)
    print(f"Top 5 pair frequencies: {sorted_pairs[:5]}")
    
    # Find best pair
    best_pair = max(pair_stats, key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Found best pair: {best_pair} with frequency: {best_freq}")
    
    # Merge the best pair
    current_splits = merge_pair(best_pair, current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"Merging {best_pair} into '{new_token}'")
    print(f"Splits after merge: {current_splits}")
    
    # Update vocabulary
    vocab.append(new_token)
    print(f"Updated vocabulary: {vocab}")
    
    # Store merge rule
    merges[best_pair] = new_token
    print(f"Updated merges: {merges}")
    
    print("-" * 30)
    
# Review final results
print("\n--- BPE merges complete ---")
print(f"Final vocabulary size: {len(vocab)}")
print("\nLearned merges (Pair -> New token):") # Print merges
for pair, token in merges.items():
    print(f"{pair} -> '{token}'")
        
print("\nFinal word splits after all merges:")
print(current_splits)
    
print("\nFinal vocabulary (sorted):")
# Sort for consistent viewing
final_vocab_sorted = sorted(list(set(vocab))) # Set removes potential duplicates
print(final_vocab_sorted)