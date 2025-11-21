import pandas as pd

# Load your existing CSV
df = pd.read_csv("raw_datasets/flores200_devtest_parallel.csv", encoding="utf-16")

# Example augmentation functions
def my_augmentation_1(text):
    return text.upper()

def my_augmentation_2(text):
    return text[::-1]  # Reversed string as a dummy example

# Dictionary mapping foil names to functions
foil_functions = {
    'uppercase': my_augmentation_1,
    'reverse': my_augmentation_2,
}

# Generate new columns for each augmentation
for foil_name, foil_func in foil_functions.items():
    new_col = f'foil_{foil_name}_eng_Latn'
    df[new_col] = df['sentence_eng_Latn'].apply(foil_func)

# Ensure "ID" column exists
if "id" not in df.columns:
    df.insert(0, "id", range(1, len(df) + 1))  # Optionally add an ID if missing

# Gather column lists
id_col = ["id"]
foil_cols = [col for col in df.columns if col.startswith('foil_')]
other_cols = [col for col in df.columns if col not in id_col + foil_cols]

# Define new column order
new_col_order = id_col + foil_cols + other_cols

# Reorder DataFrame
df = df[new_col_order]

# Save the new CSV
df.to_csv("datasets_with_foils/flores200_devtest_parallel_with_foils.csv", index=False, encoding="utf-16")

print("New CSV with foil columns sorted to the front!")
