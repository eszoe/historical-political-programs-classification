# Topic Classification of Historical Political Programs

## Overview
The goal of this group project is to train a model that classifies the content of historical party programs by political topic. As training data, we use the Habsburg Manifesto Dataset, which contains election programs from political parties active in the Austrian portion of the Habsburg Empire, covering the elections of 1897, 1900/01, 1907, and 1911. Written in German and Czech, the programs have been manually annotated into 36 topics, each with a positive and negative direction, yielding a total of 9,430 labeled sentences.

To this end, we fine-tuned ManifestoBERTa, a model developed by the Manifesto Project on the XLM-RoBERTa architecture. ManifestoBERTa classifies individual sentences from political parties' election programs into 56 political topics, drawing on a multilingual corpus of manifestos from 60 countries written in nearly 40 languages and spanning 1945 to 2023.

Furthermore, we also probed the model's transferability by applying it to historical party programs from different time periods, and a different language and political context.


## Workflow

### 1) Preparation of the raw data
- merged German and Czech data (sentences labeled by issue/topic), and their metadata (document title, year, party etc.),
- refactored variables names,
- transformed one-hot encoding to one multi-class label,
- added a dummy variable for language,
- checked fro missing data and correct variable types

### 2) Setting up train and test datasets
- identified the most frequent topics (= topics with at least 40 observations)
- removed sentences labeled with infrequent topics (less than 40 observations), as well as those tagged as "other issue" (not covered by any topic category) or "junk" (no political content)
- dataset split at the level of documents (not sentences, to avoid data leakage) for each of the two languages evenly (80/20)

### 3) Fine-tuning manifestoBERTa 
- Model 1: trained a partially frozen model (6 out of 24 layers were trained)
- Model 2: Low Rank Adaptation (LoRA) (original model weights are frozen, only training newly injected small adapter matrices)

### 4) Model evaluation
- calculated evaluation metrics (accuracy, precision macro, recall macro, macro f1, weighted f1, mcc and cross entropy) and selected the best model using macro f1, taking into account class imbalance

### 5) Examining transferability
- manually labelled four party programs from different context:
  1) The Program of the German Nationals in Graz (19.5.1870) (in German)
  2) The Program of the Christian Social Party (29.11.1926) (in German)
  3) The Appeal of the Independence and 48ers' Party to Its Voters (8.10.1896) (in Hungarian)
  4) The Program of the Reorganised Social Democratic Party of Hungary (18.04.1909) (in Hungarian)
- predicted labels using Model 1 and 2 and evaluated predictions


# Setup

1. Create a GitHub personal access token (Settings → Developer Settings → Personal Access Tokens)
2. Open the HMD_DL_Project.ipynb notebook in Google Colab
3. Click the 🔑 Secrets icon in the left sidebar
4. Add a secret named GITHUB_TOKEN with your token as the value
5. Runtime → Change runtime type → select GPU
6. Run all cells in order

# Sources
XLM-RoBERTa:
Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L. and Stoyanov, V., 2020, July. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th annual meeting of the association for computational linguistics, pp. 8440-8451. 

ManifestoBERTa: 
Burst, T., Lehmann, P., Franzmann, S., Al-Gaddooa, D., Ivanusch, C., Regel, S., Riethmüller, F., Weßels, B. and Zehnter, L., 2023. Manifestoberta (Version 56topics. sentence. 2023.1.1). 
Manifestoberta Performance Report. Sentence Version 2024a (56Topics). 2024-07-31, Available at: https://manifesto-project.wzb.eu/uploads/manifestoberta_sentence_2024_1_1_56topics_perf_report.pdf

HMD dataset:
Szöcsik, E., Zuber, C.I. and Howe, P.J., 2024. Lipset and Rokkan’s missing case: Introducing the Habsburg Manifesto Dataset. Party Politics, 30(5), pp.949-962.
