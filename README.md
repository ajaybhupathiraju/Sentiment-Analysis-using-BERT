# Sentiment-Analysis-using-BERT
BERT - Bidirectional Representation for Transformers 

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed for Natural Language Processing (NLP) tasks.
It revolutionized the way NLP models handle text by introducing bidirectional context understanding, making it highly effective for tasks like question answering, text classification, and named entity recognition.

BERT is pre-trained on large corpora of text (e.g., Wikipedia, BookCorpus) using two tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
we are using BERT to be fine-tuned on reviews of sentiment analysis task

![alt text](/bert_single_sentence_arch.png)

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Why Transformers choosen over convolution neural networks](#why-transformers-over-cnn)
- [EDA](#exploratory-data-analysis)
- [Assumptions](#assumptions)
- [Machine learning model](#machine-learning-model)
  - [Vision Transformer from scratch](#custom-vit-model)
  - [Using pretrained hugging face model](#hugging-face-pre-trained-model)
- [Loss function](#loss-function)
- [Making Predictions](#making-predictions)
- [Deployee model](#deployee-model)

## Project Overview
  
The goal of this project is to build a model using bert transformers that classifies the given input review is positive or negative.

## Data Sources
import movie reviews hugging face datasets
https://huggingface.co/datasets/stanfordnlp/imdb

refernce bert documentations from https://huggingface.co/docs/transformers/model_doc/bert
