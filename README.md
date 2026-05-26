# Topic Classification of Historical Political Programs

## Overview
The goal of this group project is to train a model that classifies the content of historical party programs into topics using a model pretrained on contemporary party programs. The resulting model could then be applied to previously unclassified historical party programs from other countries, providing a tool to researchers from social sciences and humanities.
We will use the pretrained model ManifestoBERTa, which has been trained by the Manifesto Project using the XLM-RoBERTa to classify political parties’ election programs taking sentences as units into 56 political topics. The corpus includes programs covering 60 different countries and written in almost 40 languages spanning from 1945 to 2023.
The XLM-RoBERTa is a transformer-based model trained on more than 100 languages via cross-lingual pretraining and fine-tuned with a classification head for labeling whole texts.
As our training corpus, we use the Habsburg Manifesto Dataset, which contains election programs from political parties active in the Austrian part of the Habsburg Empire, covering the elections of 1897, 1900/1, 1907, and 1911. The programs are written in German and Czech and have been manually classified into 36 topics, yielding a total of 9’430 labeled sentences.

## Workflow

### 1) Preparation of the raw data
- merged German and Czech data (sentences labeled by issue/topic), and their metadata (document title, year, party etc.),
- refactored variables names,
- transformed one-hot encoding to one multi-class label
- added a dummy variable for language

### 2) Setting up train and test datasets
- identified the most frequent topics (= topics with at least 40 observations)
- dataset split at the level of documents (not sentences, to avoid data leakage) for each of the two languages evenly (80/20)

### 3) Fine-tuning manifestoBERTa 
- Model 1: trained a partially frozen model (6 out of 24 layers were trained)
- Model 2: Low Rank Adaptation (LoRA) (original model weights are frozen, only training newly injected small adapter matrices

### 4) Model evaluation
- calculated evaluation metrics (accuracy, precision macro, recall macro, macro f1, weighted f1, mcc and cross entropy) and selected the best model using macro f1, taking into account class imbalance

### 5) Examining transferability
- manually labelled four party programs from different context:
  1) The Program of the German Nationals in Graz (19. 5.1870) (in German)
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


Szöcsik, E., Zuber, C.I. and Howe, P.J., 2024. Lipset and Rokkan’s missing case: Introducing the Habsburg Manifesto Dataset. Party Politics, 30(5), pp.949-962.
