# Topic Classification of Historical Political Programs

## Overview
The goal of this group project is to train a model that classifies the content of historical party programs into topics using a model pretrained on contemporary party programs. The resulting model could then be applied to previously unclassified historical party programs from other countries, providing a tool to researchers from social sciences and humanities.
We will use the pretrained model ManifestoBERTa, which has been trained by the Manifesto Project using the XLM-RoBERTa to classify political parties’ election programs taking sentences as units into 56 political topics. The corpus includes programs covering 60 different countries and written in almost 40 languages spanning from 1945 to 2023.
The XLM-RoBERTa is a transformer-based model trained on more than 100 languages via cross-lingual pretraining and fine-tuned with a classification head for labeling whole texts.
As our training corpus, we use the Habsburg Manifesto Dataset, which contains election programs from political parties active in the Austrian part of the Habsburg Empire, covering the elections of 1897, 1900/1, 1907, and 1911. The programs are written in German and Czech and have been manually classified into 36 topics, yielding a total of 9’430 labeled sentences

## Workflow

### 1) Data preparation
- merge German and Czech data (sentences labeled by issue/topic), and their metadata (document title, year, party etc.)  -DONE
- clean category names (DONE)
- add party names (DONE)
- identify the most frequent categories (= categories with at least 40 observations) (DONE)
- add a dummy for language

### 2) Setting up the train and the test dataset
# Even though that we a have a skewed distribution of categories, we apply the model.
- create labels
- mapping with own labels
- when splitting the dataset, the dataset shouldn't be split at the level of sentences but on the level of documents and the two languages: 
train_docs, test_docs = train_test_split(documents) otherwise: context leakage
train_sentences = expand(train_docs)
test_sentences = expand(test_docs)
- probably we need ids for documents too


# References
Burst, T., Lehmann, P., Franzmann, S., Al-Gaddooa, D., Ivanusch, C., Regel, S., Riethmüller, F., Weßels, B. and Zehnter, L., 2023. Manifestoberta (Version 56topics. sentence. 2023.1. 1).

Szöcsik, E., Zuber, C.I. and Howe, P.J., 2024. Lipset and Rokkan’s missing case: Introducing the Habsburg Manifesto Dataset. Party Politics, 30(5), pp.949-962.
