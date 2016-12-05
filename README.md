# Sentiment-Analysis
Naive Bayes Classifier for Sentiment Analysis on Movies Reviews

The purpose of this program is to create a naive Bayes classifier to perform binary sentiment analysis on movie reviews,
using Pang & Lee’s (2005) polarity dataset 2.0, which consists of 1000 positive and 1000 negative movie reviews. 
These reviews have already been pre-processed so that tokenization has already been done.
Each review is in its own file, each sentence is on its own line, and each token is followed by a space.

The classifier is trained on the first 800 files (cv0 to cv7) and is evaluated on the held-out test data: 
the 200 files beginning with cv8 and cv9.

The classifier is first trained only on 100 reviews, then 300, then 500 and finally 800 files. The purpose of this is to show how the starts with poor precision but improves over the bigger datasets.
