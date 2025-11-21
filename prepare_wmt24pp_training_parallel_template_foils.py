import os 
import ssl

# 1. Force Python's SSL system to trust everything (The "Nuclear" Option)
# This overrides the default certificate verification for the entire script.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

import pandas as pd
import nltk
import penman
import random
import amrlib
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder
from tqdm import tqdm

tqdm.pandas()

# --- 0. SETUP & LOAD MODELS ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. AMR Models (The "SentenceSmith" brains)
# STOG = Sentence to Graph (Parser)
# GTOS = Graph to Sentence (Generator)
stog_model = amrlib.load_stog_model()
gtos_model = amrlib.load_gtos_model()

# 2. NLI Model (The "Filter")
# Used to check if the meaning actually changed (Contradiction)
# Mapping for 'cross-encoder/nli-deberta-v3-small': 
# Label 0: Contradiction, Label 1: Entailment, Label 2: Neutral
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')

print(">>> MODELS LOADED.")


# --- 1. GRAPH MODIFICATION LOGIC (The "Transformations") ---

def apply_polarity_negation(triples):
    """
    [Paper Method]: Adds or removes a (:polarity -) attribute.
    Target: The root predicate or main verb.
    """
    if not triples: return triples
    
    # Heuristic: The first triple usually contains the root node of the sentence
    root_node = triples[0][0]
    
    # Check if it is already negated
    is_negated = any(t[0] == root_node and t[1] == ':polarity' for t in triples)
    
    new_triples = list(triples)
    if is_negated:
        # REMOVE negation (Double negative -> Positive)
        new_triples = [t for t in new_triples if not (t[0] == root_node and t[1] == ':polarity')]
    else:
        # ADD negation
        new_triples.append((root_node, ':polarity', '-'))
        
    return new_triples

def apply_role_swap(triples):
    """
    [Paper Method]: Swaps :ARG0 (Agent) and :ARG1 (Patient).
    Example: "Tiger bites Snake" -> "Snake bites Tiger"
    """
    # Map nodes to their arguments: node -> {':ARG0': 'x', ':ARG1': 'y'}
    candidates = {}
    
    for src, rel, tgt in triples:
        if rel in [':ARG0', ':ARG1']:
            if src not in candidates: candidates[src] = {}
            candidates[src][rel] = tgt
            
    # Find nodes that have BOTH an Agent (ARG0) and Patient (ARG1)
    swap_targets = [node for node, args in candidates.items() if ':ARG0' in args and ':ARG1' in args]
    
    if not swap_targets:
        return triples # Cannot swap if ingredients are missing
    
    # Pick one event to swap (randomly if there are multiple verbs)
    target_node = random.choice(swap_targets)
    arg0_val = candidates[target_node][':ARG0']
    arg1_val = candidates[target_node][':ARG1']
    
    # Build new triples list with swapped targets
    new_triples = []
    for src, rel, tgt in triples:
        if src == target_node:
            if rel == ':ARG0':
                new_triples.append((src, rel, arg1_val)) # ARG0 takes ARG1's value
            elif rel == ':ARG1':
                new_triples.append((src, rel, arg0_val)) # ARG1 takes ARG0's value
            else:
                new_triples.append((src, rel, tgt))
        else:
            new_triples.append((src, rel, tgt))
            
    return new_triples


# --- 2. FILTERING LOGIC (Cleanup & Quality Control) ---

def post_process_text(original, foil):
    """Basic text cleanup (capitalization, punctuation)."""
    if not foil: return None
    
    # Match Capitalization of first letter
    if original and original[0].isupper() and foil:
        foil = foil[0].upper() + foil[1:]
        
    # Fix common AMR generation artifacts like spaces before punctuation
    foil = re.sub(r'\s+([.,;!?])', r'\1', foil)
    return foil

def passes_filters(original, foil, modification_type):
    """
    Returns True if the foil is good, False if it's garbage.
    Combines Heuristics and NLI check.
    """
    if not foil: return False
    
    # A. IDENTITY CHECK (Did the generator ignore us?)
    # Remove spaces/punctuation/case to compare "core" text
    def clean(s): return re.sub(r'[^a-z0-9]', '', s.lower())
    if clean(original) == clean(foil):
        return False

    # B. LENGTH CHECK (Did the parser crash or hallucinate?)
    len_ratio = len(foil) / max(1, len(original))
    if len_ratio < 0.5 or len_ratio > 1.5:
        return False

    # C. REPETITION CHECK (Did the generator stutter?)
    words = foil.split()
    if len(words) > 10:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) != len(set(trigrams)):
            return False

    # D. NLI SEMANTIC CHECK (The "Brain" Check)
    # We want to avoid Entailment (meaning the same thing).
    # We verify the foil actually contradicts or changes meaning.
    scores = nli_model.predict([(original, foil)])
    pred_label = scores.argmax()
    
    # Labels: 0=Contradiction, 1=Entailment, 2=Neutral
    if pred_label == 1: # Entailment
        return False # Fail: Meaning didn't change enough
        
    return True


# --- 3. CORE PIPELINE FUNCTION ---

def run_sentencesmith_direct(sentence, modification_type):
    try:
        # 1. PARSE
        graphs = stog_model.parse_sents([sentence])
        graph_string = graphs[0]
        if isinstance(graph_string, list):
            graph_string = graph_string[0]
        
        # print(f'the type of graph_string is {type(graph_string)}')
        # 2. DECODE
        g = penman.decode(graph_string)
        triples = g.triples 
        
        # 3. MODIFY
        if modification_type == "polarity_negation":
            mod_triples = apply_polarity_negation(triples)
        elif modification_type == "RS":
            mod_triples = apply_role_swap(triples)
        else:
            mod_triples = triples 
        
        # DEBUG: Check if triples actually changed
        if mod_triples == triples:
            print(f"   [FAIL] No changes made to graph for: '{sentence[:20]}...'")
            return None

        # 4. ENCODE & GENERATE
        new_g = penman.Graph(mod_triples)
        new_graph_string = penman.encode(new_g)
        sent_list = gtos_model.generate([new_graph_string])
        # print(f'the sent list is {sent_list}')
        raw_foil = sent_list[0][0]
        # print(f'the type of raw_foil is {type(raw_foil)} {raw_foil}')
        
        # 5. CLEANUP & FILTER
        clean_foil = post_process_text(sentence, raw_foil)
        # print(f'the type of clean_foil is {type(clean_foil)} {clean_foil}')
        
        # DEBUG: Print the rejection reason
        if not clean_foil:
            print(f"   [FAIL] Empty generation for: '{sentence[:20]}...'")
            return None
            
        # Identity Check
        def clean(s): return re.sub(r'[^a-z0-9]', '', s.lower())
        if clean(sentence) == clean(clean_foil):
            print(f"   [FAIL] Identity Filter (Text didn't change): '{clean_foil}'")
            return None

        # # NLI Check
        # scores = nli_model.predict([(sentence, clean_foil)])
        # pred_label = scores.argmax()
        # if pred_label == 1: # Entailment
        #     print(f"   [FAIL] NLI Filter (Meaning is too similar): '{clean_foil}'")
        #     return None
            
        # SUCCESS
        print(f"   [SUCCESS] Created Foil: '{clean_foil}'")
        return clean_foil

    except Exception as e:
        print(f"   [ERROR] Crash on '{sentence[:20]}...': {e}")
        return None


# --- 4. SCORERS & WRAPPERS (Paragraph Level) ---

def eval_suitability_for_PN(sent):
    """Score suitability for Polarity Negation (needs verbs)."""
    score = 0.1
    # Check for auxiliary verbs
    if any(w in sent.lower() for w in [' is ', ' are ', ' was ', ' were ', ' can ', ' should ']):
        score += 0.5
    # Check length
    length = len(sent.split())
    if 10 <= length <= 40: score += 0.4
    return score

def eval_suitability_for_RS(sent):
    """Score suitability for Role Swap (needs two entities + verb)."""
    score = 0.1
    # 'by' often implies passive voice (Target by Agent), ideal for swapping
    if " by " in sent: score += 0.8
    if len(sent.split()) > 12: score += 0.2
    return score

def min_length_chars(min_length):
    def filter_fn(sent): return len(sent) >= min_length
    return filter_fn

def foil_augment_paragraph(text, pipeline):
    """
    Walks through a paragraph, finds the best sentence, creates a foil, 
    and injects it back into the paragraph.
    """
    text_str = str(text)
    if not text_str or text_str.lower() == 'nan': return text, None

    sents = sent_tokenize(text_str)
    # Pre-filter sentences (must be long enough)
    candidates = [sent for sent in sents if pipeline.get("filter", lambda x: True)(sent)]
    
    if not candidates: return text, None

    # Score sentences to find the best target
    scores = [pipeline["evaluate"](sent) for sent in candidates]
    if not scores: return text, None
    
    best_idx = scores.index(max(scores))
    orig_sent = candidates[best_idx]
    
    # --- GENERATE FOIL ---
    augmented_sent = pipeline["augment"](orig_sent)
    
    # If generation failed or was filtered out, return original text (no change)
    if augmented_sent is None:
        return text, None 
        
    # --- INJECT BACK INTO PARAGRAPH ---
    position = None
    if pipeline.get("measure_position", False):
        idx = text_str.find(orig_sent)
        if idx >= 0:
            position = idx / max(1, len(text_str))
            
    # Replace only the first occurrence of the target sentence
    replaced_text = text_str.replace(orig_sent, augmented_sent, 1)
    return replaced_text, position


# --- 5. CONFIGURATION ---

foil_pipelines = {
    "polarity_negation": {
        "evaluate": eval_suitability_for_PN,
        "augment": lambda s: run_sentencesmith_direct(s, "polarity_negation"),
        "filter": min_length_chars(15),
        "measure_position": True,
    },
    "role_swap": {
        "evaluate": eval_suitability_for_RS,
        "augment": lambda s: run_sentencesmith_direct(s, "RS"),
        "filter": min_length_chars(10),
        "measure_position": True,
    },
}


# --- 6. MAIN EXECUTION ---

input_file = "raw_datasets/wmt24pp_parallel.csv" 
output_file = "datasets_with_foils/wmt24pp_parallel_foils.csv"

print(f">>> Reading {input_file}...")
df = pd.read_csv(input_file, encoding="utf-16")


# Process each pipeline
for foil_name, pipeline in foil_pipelines.items():
    col_text = f'foil_{foil_name}_eng_Latn'
    col_pos  = f'foil_{foil_name}_eng_Latn_RELATIVE_POSITION'
    
    print(f">>> Generating Foils for: {foil_name} ...")
    
    # Apply the logic to the English column (en_EN)
    # Returns a tuple (new_text, position)
    results = df['en_EN'].progress_apply(lambda x: foil_augment_paragraph(x, pipeline))
    
    # Unpack results
    df[col_text] = results.apply(lambda x: x[0])
    
    if pipeline.get("measure_position", False):
        df[col_pos] = results.apply(lambda x: x[1])

# Add ID column if missing
ID_COLUMN = "segment_id"
if ID_COLUMN not in df.columns:
    df.insert(0, ID_COLUMN, range(1, len(df) + 1))

# Reorder columns for cleanliness
id_col = [ID_COLUMN]
foil_cols = [col for col in df.columns if col.startswith('foil_')]
other_cols = [col for col in df.columns if col not in id_col + foil_cols]
new_col_order = id_col + foil_cols + other_cols
df = df[new_col_order]

# Save
print(f">>> Saving to {output_file}...")
df.to_csv(output_file, index=False, encoding="utf-16") # Keeping utf-16 to match input
print(">>> Done.")