# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 16:05:22 2015

@author: dimit_000
"""

from collections import defaultdict
import math
import os
from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
from nltk.stem.lancaster import LancasterStemmer


class NaiveBayes:
    '''
    Constructs a baseline naive Bayes classifier that uses only tokenized words as features,
    that is equivalent to a unigram language model for each class (positive and negative).
    '''
    
    def __init__(self, docs, *args):
        '''
        Initialize the classifier
        Parameters:
            docs: dictionary contatining positive and negative reviews
            *args: optional list of substrings that if the documents contain in their name, they'll included in the training set
        '''
        
        # Seperate the positive and the negative reviews
        self.neg = docs['negative']
        self.pos = docs['positive']
        
        # Create the positive and negative training sets if they include the title of the args
        self.trainset_pos = {k:self.pos[k] for k in self.pos.keys() for title in args if title in k}
        self.trainset_neg = {k:self.neg[k] for k in self.neg.keys() for title in args if title in k}
        
            
    def training(self, documents):
        '''
        Compute the maximum likelihood estimation of the model
        '''
    # Initialize the training dictionary and the count of tokens
    	freq = {}
    	words_len = 0
        # Navigate through the training reviews and get the tokens	
        for doc_name in documents.keys():
    		for word in documents[doc_name]:
             # Keep a count of how many tokens are in the training reviews
    			words_len += 1
    			# Form dictionary that holds the frequency of each token in the training set
    			try: 
    				freq[word] += 1
    			except: 
    				freq[word] = 1
    	
    # How many tokens are in the training set
    	vocabulary_len = len(freq)
    
    	for word, value in freq.items():
         # Add Laplace smoothing for the counts of the tokens
    		value += 1
         # Count the frequency of each token in the training set, and log the result for the maximum likelihood estimation
         # Note the denominator is twice the the vocabulary length, since I've added 1 to every token count
    		freq[word] = math.log(value / float(vocabulary_len + vocabulary_len), 10)
    	
    	return  freq
     

    def predict(self, freq_pos, freq_neg, *args):
        '''
        Predict the class (positive or negative), given a validation set of reviews
        '''
        # 
        self.pos_preds = []
        self.neg_preds= []
        
        # Seperate the validation set to two according to the sentiment
        self.val_set = [{k: self.pos[k] for k in self.pos.keys() for title in args if title in k}, 
                         {k: self.neg[k] for k in self.neg.keys() for title in args if title in k}]
        
        # Loop through the validation set, each sentiment every time
        for sentiment in self.val_set: 
            # Loop through the reviews in each class of sentiment
            for doc_name in sentiment.keys():
                pos_prob = 0
                neg_prob = 0
                # Loop through the tokens in every review
                for word in sentiment[doc_name]:
                    # If the token is positive, add that to the positive likelihood, otherwise to the negative
                    try:
                        pos_prob += freq_pos[word]
                    except:
                        pass
                    try:
                        neg_prob += freq_neg[word]
                    except:
                        pass
                # Predict the sentiment of each document depending on the likelihood of the tokens
                if pos_prob > neg_prob:
                    self.pos_preds.append(doc_name)
                else:
                    self.neg_preds.append(doc_name)
 

    def confusMatrix(self):
        '''
        Compute the confusion matrix for the predictions
        '''
        self.val_pos, self.val_neg = self.val_set
        
        # Get the count of true positives (predicted positive reviews that are actual positives)
        pos_tp = 0
        for pos_pred in self.pos_preds:
            if pos_pred in self.val_pos.keys():
                pos_tp += 1
        
        # Get the count of false positives (predicted positive reviews that are not actual positives)
        pos_fp = 0
        for pos_pred in self.pos_preds:
    		if pos_pred in self.val_neg.keys():
    			pos_fp += 1    
        
        # Get the count of negative true positives (predicted negative reviews that are actual negatives)
        neg_tp = 0
        for neg_pred in self.neg_preds:
    		if neg_pred in self.val_neg.keys():
    			neg_tp += 1
       
       # Get the count of negative false positives (predicted negative reviews that are actual positives)
        neg_fp = 0
        for neg_pred in self.neg_preds:
    		if neg_pred in self.val_pos.keys():
    			neg_fp += 1
    
        return (pos_tp, pos_fp, neg_tp, neg_fp)
    
    
    def bayesPerformance(self, confusion):
        '''
        Compute precision recall and F statistic
        '''
        tp, fp, tn, fn = confusion    
        
        precision = tp / float(tp + fp)
        recall = tp / float (tp + fn)
        F = (2 * precision * recall) / float(precision + recall)
    
        return (precision,recall,F)
    
    
# main()

# This will include all the reviews	
docs = {}
# There are two folders, one containing the positive movie reviews and another the negative reviews
paths = ['yourPathHere\\txt_sentoken\\pos']
sentiments = ['negative', 'positive']

####################
# Read in the data #
####################

# The list of stopwords to exclude from the reviews
stop=stopwords.words('english')

# Use the Lancaster Stemmer from ntlk, which is more aggressive than the Porter Stemmer
stemmer = LancasterStemmer()

# Loop through the negative and positive reviews in order
for i in range(2):
    path = paths[i]
    sent = sentiments[i]
    # Each doc contains the filenames as keys and the tokens of the reviews as values
    doc = defaultdict(list)
    os.chdir(path)
    filenames = os.listdir(path)   
    # Navigate through the reviews of the corresponding sentiment (good or bad)
    for files in filenames:
        with open(files) as f:
            words = []
            # Loop through the lines of each review to add the tokens to the dictionary
            for line in f:
                # Add the token to the list only if it's a word
                words.extend( [ stemmer.stem( word.strip() ) for word in line.split(' ') \
                if word.lower() not in stop and word.lower() not in ('!','.',':',',','(',')',';','"','*','--','-','?','|','$') \
                and not word.lower().isdigit() and word and word.strip() ] )
            # Add the list of tokens to the corresponding file
            doc[files] = words
    # The final data structure will be a dictionary containing two keys, one for each list of positive and negative reviews
    docs[sent] = doc
        

# Initialize a new Naive Bayes classifier, and pass on the training set        
classifier100 = NaiveBayes(docs, 'cv0')

# Train the model to learn the parameters
freq_pos100 = classifier100.training(classifier100.trainset_pos)
freq_neg100 = classifier100.training(classifier100.trainset_neg)

# Predict on the validation set
classifier100.predict(freq_pos100, freq_neg100, 'cv6', 'cv7')

# Obtain the confusion matrix
conf100 = classifier100.confusMatrix()

# Copmute precision, recall and F statistic
precision100, recall100, F100 = classifier100.bayesPerformance(conf100)



# Repeat the excercise over 300 training documents
classifier300 = NaiveBayes(docs, 'cv0', 'cv1', 'cv2')

freq_pos300 = classifier300.training(classifier300.trainset_pos)
freq_neg300 = classifier300.training(classifier300.trainset_neg)

classifier300.predict( freq_pos300, freq_neg300, 'cv6', 'cv7' )

conf300 = classifier300.confusMatrix()

precision300, recall300, F300 = classifier300.bayesPerformance(conf300)


# Repeat the excercise over 500 training documents
classifier500 = NaiveBayes(docs, 'cv0', 'cv1', 'cv2', 'cv3', 'cv4')

freq_pos500 = classifier500.training(classifier500.trainset_pos)
freq_neg500 = classifier500.training(classifier500.trainset_neg)

classifier500.predict( freq_pos500, freq_neg500, 'cv6', 'cv7' )

conf500 = classifier500.confusMatrix()

precision500, recall500, F500 = classifier500.bayesPerformance(conf500)



# Repeat the excercise over 800 training documents
classifier800 = NaiveBayes(docs, 'cv0', 'cv1', 'cv2', 'cv3', 'cv4', 'cv5', 'cv6', 'cv7')

freq_pos800 = classifier800.training(classifier800.trainset_pos)
freq_neg800 = classifier800.training(classifier800.trainset_neg)

classifier800.predict( freq_pos800, freq_neg800, 'cv8', 'cv9' )

conf800 = classifier800.confusMatrix()

precision800, recall800, F800 = classifier800.bayesPerformance(conf800)


# Plot the F accuracy of the models trained on the different datasets
import matplotlib.pyplot as plt
plt.plot([F100,F300,F500,F800])
plt.ylabel('F accuracy over expanding training sets')
plt.show()
