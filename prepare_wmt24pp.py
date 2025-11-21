import pandas as pd
import nltk

# Download NLTK sentence tokenizer resources (only needs to run once)
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# -------------------
# Parameterized min length filter
def min_length_chars(min_length):
    def filter_fn(sent):
        return len(sent) >= min_length
    return filter_fn

# Example evaluation/scoring functions (replace these with your real criteria)
def eval_suitability_for_reverse_foil(sent):
    return len(sent) / 100
def eval_suitability_for_upper_foil(sent):
    return 1

def eval_first(sent):
    return 1.0

# Example augmentation functions (replace with your real augmentation logic)
def aug_reverse(sent):
    return sent[::-1]
def aug_upper(sent):
    return sent.upper()

# -------------------
# Define foil pipelines: add "measure_position" key!
foil_pipelines = {
    "length_reverse": {
        "evaluate": eval_suitability_for_reverse_foil,
        "augment": aug_reverse,
        "filter": min_length_chars(35),
        "measure_position": True,    # Position measurement ON for this one
    },
    "first_upper": {
        "evaluate": eval_suitability_for_upper_foil,
        "augment": aug_upper,
        "filter": min_length_chars(35),
        "measure_position": True,    # Turn ON or OFF as needed per pipeline
    },
}

# -------------------
def foil_augment_text(text, pipeline):
    """
    Applies the foil pipeline and (optionally) computes the relative position
    (0=start, 1=end) of the original sentence in the original text.
    Returns (new_text, position) if position measurement is ON,
    otherwise just new_text.
    """
    sents = sent_tokenize(str(text))
    if not sents:
        return text, None
    # Filter sentences based on the filter function (if provided)
    candidates = [sent for sent in sents if pipeline.get("filter", lambda x: True)(sent)]
    if not candidates:
        return text, None
    # Evaluate and select the best sentence
    scores = [pipeline["evaluate"](sent) for sent in candidates]
    max_score = max(scores)
    best_idx = scores.index(max_score)
    orig_sent = candidates[best_idx]
    augmented_sent = pipeline["augment"](orig_sent)
    # Compute relative position if needed
    position = None
    if pipeline.get("measure_position", False):
        idx = str(text).find(orig_sent)
        if idx >= 0:
            position = idx / max(1, len(str(text)))  # 0 to 1 scale, avoids division by zero
    # Replace only the first occurrence of the selected sentence in the original text
    replaced_text = str(text).replace(orig_sent, augmented_sent, 1)
    return replaced_text, position

# -------------------
# Load the input CSV
df = pd.read_csv("raw_datasets/wmt24pp_parallel.csv", encoding="utf-16")

# For each foil pipeline, create new columns with augmented text and (optionally) position
for foil_name, pipeline in foil_pipelines.items():
    col_text = f'foil_{foil_name}_eng_Latn'
    col_pos  = f'foil_{foil_name}_eng_Latn_RELATIVE_POSITION'
    print(f"Generating {col_text}...")
    results = df['en_EN'].apply(lambda x: foil_augment_text(x, pipeline))
    # Always create the text column
    df[col_text] = results.apply(lambda x: x[0])
    # Optionally create the position column if measure_position is ON
    if pipeline.get("measure_position", False):
        df[col_pos] = results.apply(lambda x: x[1])

ID_COLUMN = "segment_id"
# If "id" column does not exist, create it (starting from 1)
if ID_COLUMN not in df.columns:
    df.insert(0, ID_COLUMN, range(1, len(df) + 1))

# Reorder columns: id first, then all foil columns (including position), then the rest
id_col = [ID_COLUMN]
foil_cols = [col for col in df.columns if col.startswith('foil_')]
other_cols = [col for col in df.columns if col not in id_col + foil_cols]
new_col_order = id_col + foil_cols + other_cols
df = df[new_col_order]

# Save the new DataFrame to CSV
output_path = "datasets_with_foils/wmt24pp_parallel.csv"
df.to_csv(output_path, index=False, encoding="utf-16")
print(f"New CSV with foil columns and (optionally) position saved as {output_path}")