import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

corpus = pd.read_csv(DATA_DIR / 'corpus_ger_cze.csv')

# Categories with more than 40 observations
sums = corpus.columns[5:].tolist()
sums = corpus[sums].sum().astype(int)

filtered = sums[
    (sums >= 40)
].sort_values(ascending=False)

result = filtered.reset_index()
result.columns = ['Category', 'Count']

print("Category\tCount")
for _, row in result.iterrows():
    print(f"{row['Category']}\t{row['Count']}")


# Select only directional categories with more than 40 observations
sums = corpus.columns[4:64].tolist()
sums = corpus[sums].sum().astype(int)

filtered = sums[
    (sums >= 40)
].sort_values(ascending=False)

result = filtered.reset_index()
result.columns = ['Category', 'Count']

print("Category\tCount")
for _, row in result.iterrows():
    print(f"{row['Category']}\t{row['Count']}")


# Barplot of category frequencies of positional categories
plt.figure(figsize=(10, 8))

plt.barh(result['Category'], result['Count'])

plt.xlabel('Count')
plt.ylabel('Category')
plt.title('Category Counts (>= 40)', fontsize=16)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig('category_counts.png', dpi=300, bbox_inches='tight')
plt.show()

