from datasets import load_dataset
import pandas as pd

# Load the 'devtest' split with all languages
dataset = load_dataset("Muennighoff/flores200", "all", split="devtest")

# Convert to pandas DataFrame (this may take a minute)
df = pd.DataFrame(dataset)

# Save the full DataFrame as CSV
df.to_csv("flores200_devtest_parallel.csv", index=False)

print("CSV created! Check your current directory for 'flores200_devtest_parallel.csv'")