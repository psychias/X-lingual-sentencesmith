from datasets import load_dataset, get_dataset_config_names
import pandas as pd

# Get all EN-XX configs
all_pairs = get_dataset_config_names("google/wmt24pp")
all_pairs = [lp for lp in all_pairs if lp.startswith("en-")]

# Initialize base DataFrame
base_df = None

for idx, lp in enumerate(all_pairs):
    print(f"Loading {lp}...")
    ds = load_dataset("google/wmt24pp", lp, split="train")
    df = pd.DataFrame(ds)
    # Rename columns
    df = df.rename(columns={"source": "en_EN"})
    lang_code = lp.split("-", 1)[1]
    df = df.rename(columns={"target": lang_code})
    # Drop the 'lp' column if it exists
    if "lp" in df.columns:
        df = df.drop(columns=["lp"])
    # Set up base DataFrame
    if base_df is None:
        base_df = df
    else:
        # Only bring in segment_id and new language column
        join_cols = ["segment_id"]
        # If you want to be more robust (e.g. segment_id + domain + ...), change here:
        for col in ['document_id', 'domain', 'is_bad_source']:  # adjust as needed
            if col in df.columns and col in base_df.columns:
                join_cols.append(col)
        base_df = base_df.merge(
            df[["segment_id", lang_code]], 
            on="segment_id",  # ONLY ON segment_id to avoid dups
            how="inner"
        )

# Delete "original_target" if it exists
if "original_target" in base_df.columns:
    base_df = base_df.drop(columns=["original_target"])
print("Saving merged file...")

# Save as UTF-16
base_df.to_csv("raw_datasets/wmt24pp_parallel.csv", index=False, encoding="utf-16")
print("Saved: wmt24pp_parallel.csv (UTF-16)")