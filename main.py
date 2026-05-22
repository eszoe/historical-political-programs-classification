from pathlib import Path
from data_prep_load import load_corpus
from data_prep_recode import recode_corpus
from label_counts import get_frequent_label_counts, plot_label_counts


corpus = load_corpus(Path('data'))

corpus_rec = recode_corpus(corpus)
print(f"Original corpus size: {len(corpus_rec)} sentences")

# Display labels with at least 40 observations
freq_labels = get_frequent_label_counts(corpus_rec, min_count=40)
print(freq_labels.to_string(index=False))
print(f"Labels above threshold: {len(freq_labels)}")
plot_label_counts(freq_labels, min_count=40)

# Display dataset size if using only labels with at least 40 observations
filtered_size = freq_labels['Count'].sum()
print(f"Corpus size if ONLY keeping frequent labels: {filtered_size} sentences")