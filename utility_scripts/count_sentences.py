import pandas as pd
import nltk
import spacy

# Ensure required resources
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

nlp = spacy.load("en_core_web_sm")

# Load your file
df = pd.read_csv("raw_datasets/wmt24pp_parallel.csv", encoding="utf-16")

# Functions to count and split
def nltk_split(text):
    return sent_tokenize(str(text))

def spacy_split(text):
    doc = nlp(str(text))
    return [sent.text for sent in doc.sents]

# Count sentences
df['num_sentences_nltk'] = df['en_EN'].apply(lambda x: len(nltk_split(x)))
df['num_sentences_spacy'] = df['en_EN'].apply(lambda x: len(spacy_split(x)))

# Add segmented sentences for display
df['nltk_sents'] = df['en_EN'].apply(nltk_split)
df['spacy_sents'] = df['en_EN'].apply(spacy_split)

# Stats
print("\nSummary stats for NLTK:")
print(df['num_sentences_nltk'].describe())
print("\nSummary stats for spaCy:")
print(df['num_sentences_spacy'].describe())

print("\nValue counts for NLTK:")
print(df['num_sentences_nltk'].value_counts().sort_index())
print("\nValue counts for spaCy:")
print(df['num_sentences_spacy'].value_counts().sort_index())

# Find rows where the methods disagree
disagree = df[df['num_sentences_nltk'] != df['num_sentences_spacy']]
print(f"\nExamples where NLTK and spaCy disagree: {disagree.shape[0]}")

# Show a few comparative examples
print("\nA few comparative examples where NLTK and spaCy disagree:")
for idx, row in disagree.head(5).iterrows():
    print("="*60)
    print(f"segment_id: {row['segment_id']}")
    print(f"Original en_EN: {row['en_EN']}")
    print(f"NLTK ({row['num_sentences_nltk']}): {row['nltk_sents']}")
    print(f"spaCy ({row['num_sentences_spacy']}): {row['spacy_sents']}")
    print()
