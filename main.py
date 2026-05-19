from pathlib import Path
from data_preparation import load_corpus
from recode_variables import recode_corpus
from overview_labels import get_category_counts, plot_category_counts

corpus = load_corpus(Path('data'))

corpus_rec = recode_corpus(corpus)

result = get_category_counts(corpus_rec, min_count=40)

plot_category_counts(result, min_count=40)

print(corpus_rec.head())