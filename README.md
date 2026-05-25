# Topic Classification of Historical Political Programs

## Overview
The goal of this group project is to train a model that classifies the content of historical party programs into topics using a model pretrained on contemporary party programs. The resulting model could then be applied to previously unclassified historical party programs from other countries, providing a tool to researchers from social sciences and humanities.
We will use the pretrained model ManifestoBERTa, which has been trained by the Manifesto Project using the XLM-RoBERTa to classify political parties’ election programs taking sentences as units into 56 political topics. The corpus includes programs covering 60 different countries and written in almost 40 languages spanning from 1945 to 2023.
The XLM-RoBERTa is a transformer-based model trained on more than 100 languages via cross-lingual pretraining and fine-tuned with a classification head for labeling whole texts.
As our training corpus, we use the Habsburg Manifesto Dataset, which contains election programs from political parties active in the Austrian part of the Habsburg Empire, covering the elections of 1897, 1900/1, 1907, and 1911. The programs are written in German and Czech and have been manually classified into 36 topics, yielding a total of 9’430 labeled sentences.

## Workflow

### 1) Data preparation
- merged German and Czech data (sentences labeled by issue/topic), and their metadata (document title, year, party etc.)
- cleaned topic names
- identified the most frequent topics (= topics with at least 40 observations)
- added a dummy variable for language

### 2) Setting up train and test datasets
- dataset split at the level of documents (not sentences, to avoid data leakage) for each of the two languages evenly

# Setup

1. Create a GitHub personal access token (Settings → Developer Settings → Personal Access Tokens)
2. Open the HMD_DL_Project.ipynb notebook in Google Colab
3. Click the 🔑 Secrets icon in the left sidebar
4. Add a secret named GITHUB_TOKEN with your token as the value
5. Runtime → Change runtime type → select GPU
6. Run all cells in order

# References
Burst, T., Lehmann, P., Franzmann, S., Al-Gaddooa, D., Ivanusch, C., Regel, S., Riethmüller, F., Weßels, B. and Zehnter, L., 2023. Manifestoberta (Version 56topics. sentence. 2023.1. 1).

Szöcsik, E., Zuber, C.I. and Howe, P.J., 2024. Lipset and Rokkan’s missing case: Introducing the Habsburg Manifesto Dataset. Party Politics, 30(5), pp.949-962.
