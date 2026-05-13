import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path('data')
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)  # create if doesn't exist

corpus_rec = pd.read_csv(DATA_DIR / 'corpus_rec.csv')


def get_category_counts(corpus: pd.DataFrame, min_count: int = 40) -> pd.DataFrame:
    """
    Selects directional category columns (_pos/_neg),
    sums their occurrences, and filters by minimum count.

    Args:
        corpus:    the full dataframe
        min_count: minimum number of observations to keep a category

    Returns:
        DataFrame with columns ['Category', 'Count']
    """
    # Select columns by name pattern — robust to column order changes
    category_cols = [
        col for col in corpus.columns
        if col.endswith('_pos') or col.endswith('_neg')
    ]

    counts = corpus[category_cols].sum().astype(int)

    filtered = (
        counts[counts >= min_count]
        .sort_values(ascending=False)
        .reset_index()
    )
    filtered.columns = ['Category', 'Count']

    return filtered


def plot_category_counts(result: pd.DataFrame, min_count: int = 40):
    """
    Plots a horizontal bar chart of category frequencies.
    """
    plt.figure(figsize=(10, 8))
    plt.barh(result['Category'], result['Count'])
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.title(f'Category Counts (>= {min_count})', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_path = PLOTS_DIR / 'category_counts.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


# Run
result = get_category_counts(corpus_rec, min_count=40)
print(result.to_string(index=False))
plot_category_counts(result, min_count=40)