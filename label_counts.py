import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PLOTS_DIR = Path('plots')


def get_frequent_label_counts(corpus: pd.DataFrame, min_count: int = 40) -> pd.DataFrame:
    """
    Selects directional topic columns (_pos/_neg),
    sums their occurrences, and filters by minimum count.

    Args:
        corpus:    the full dataframe
        min_count: minimum number of observations to keep a topic

    Returns:
        DataFrame with columns ['Topic', 'Count']
    """
    # Select columns by name pattern
    topic_cols = [
        col for col in corpus.columns
        if col.endswith('_pos') or col.endswith('_neg')
    ]

    counts = corpus[topic_cols].sum().astype(int)

    filtered = (
        counts[counts >= min_count]
        .sort_values(ascending=False)
        .reset_index()
    )
    filtered.columns = ['Topic', 'Count']

    return filtered


def plot_label_counts(label_counts: pd.DataFrame, min_count: int = 40):
    """
    Plots a horizontal bar chart of topic frequencies.

    Args:
        label_counts: DataFrame with columns ['Topic', 'Count'],
                      already filtered by minimum count
        min_count: used only for the plot title — filtering
                   should be done upstream in get_frequent_label_counts()
    """
    PLOTS_DIR.mkdir(exist_ok=True)  # create if doesn't exist

    plt.figure(figsize=(10, 8))
    plt.barh(label_counts['Topic'], label_counts['Count'])
    plt.xlabel('Count')
    plt.ylabel('Topic')
    plt.title(f'Topic Counts (>= {min_count})', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_path = PLOTS_DIR / 'label_counts.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    from data_prep_load import load_corpus
    from data_prep_recode import recode_corpus

    corpus = load_corpus(Path('data'))
    corpus_rec = recode_corpus(corpus)

    freq_labels = get_frequent_label_counts(corpus_rec, min_count=40)
    print(freq_labels.to_string(index=False))
    plot_label_counts(freq_labels, min_count=40)