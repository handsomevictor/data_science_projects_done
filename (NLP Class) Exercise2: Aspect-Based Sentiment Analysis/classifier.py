# -*- coding: utf-8 -*-
# Importing libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
import torch
import transformers as ppb
import pickle

# Loading pre-trained model
# Here we are using DistilBERT Model
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# We can also use BERT instead of DistilBERT then need to uncomment the below line
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights) #.to('cuda')

class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile, devfile=None):
        """Trains the classifier model on the training set stored in file trainfile"""
        # Reading the data and loading into a dataframe
        df = pd.read_csv(trainfile, delimiter='\t', header=None, names=["polarity", "aspect_category", "aspect_term", "position", "sentence"])

        "Tokenization"
        # Tokenize the sentences - break them up into word and subwords 
        tokenized = df['sentence'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

        # Finding the length of the largest sentence
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        "Padding"
        # To represent the input as one 2-d array pad all lists to the same size
        padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])

        "Masking"
        # Masking helps in ignoring the padding while processing the input
        attention_mask = np.where(padded != 0, 1, 0)

        # Converting 'pad' and 'mask' into tensor
        input_ids = torch.tensor(padded)#.to('cuda')
        attention_mask = torch.tensor(attention_mask)#.to('cuda')

        # Running our data (sentences) through BERT model
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        # Saving the 'features' for logistics regression model
        features = last_hidden_states[0][:,0,:].cpu().numpy()

        # Saving the 'polarity' of the sentence to the 'labels'
        labels = df['polarity']

        "Train"
        # Training the model
        
        #clf = LogisticRegression(max_iter=2000) #Acc.: 81.65
        #clf = LogisticRegression(penalty='l2', C=5.26, solver='liblinear', max_iter=2000) #Acc.: 80.59
        #clf = LogisticRegression(C=5.26, max_iter=2000) #Acc.: 79.26
        #clf = LogisticRegression(solver='liblinear', max_iter=2000) #Acc.: 80.85
        #clf = LogisticRegression(solver='newton-cg', max_iter=2000) #Acc.: 81.65
        clf = LogisticRegression(C=0.1, max_iter=2000) #Acc.: 82.45
        #clf = LogisticRegression(C=0.0001, max_iter=2000) #Acc.: 70.21
        #clf = LogisticRegression(C=0.01, max_iter=2000) #Acc.: 81.91
        #clf = LogisticRegression(C=0.5, max_iter=2000) #Acc.: 81.38

        #clf = SVC(gamma='auto') #Acc.: 70.21
        #clf = RandomForestClassifier(n_estimators=128) #with 64 trees, mean acc of 79.36; 128 trees, mean acc of 80.16
        # clf = OneVsRestClassifier(LogisticRegression(max_iter=2000)) #Acc.: 80.85

        clf.fit(features, labels)

        # Saving weights of the model
        filename = 'final_model.h5'
        pickle.dump(clf, open(filename, 'wb'))
        print('Training is finished')

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        # Reading the data and loading into a dataframe
        df_test = pd.read_csv(datafile, delimiter='\t', header=None, names=["polarity", "aspect_category", "aspect_term", "position", "sentence"])

        "Tokenization"
        # Tokenize the sentences - break them up into word and subwords 
        tokenized_test = df_test['sentence'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

        # Finding the length of the largest sentence
        max_len = 0
        for i in tokenized_test.values:
            if len(i) > max_len:
                max_len = len(i)

        "Padding"
        # To represent the input as one 2-d array pad all lists to the same size
        padded_test = np.array([i + [0]*(max_len - len(i)) for i in tokenized_test.values])

        "Masking"
        # Masking helps in ignoring the padding while processing the input
        attention_mask_test = np.where(padded_test != 0, 1, 0)

        # Converting 'pad' and 'mask' into tensor
        input_ids_test = torch.tensor(padded_test)#.to('cuda')
        attention_mask_test = torch.tensor(attention_mask_test)#.to('cuda')

        # Running our data (sentences) through BERT model
        with torch.no_grad():
            last_hidden_states_test = model(input_ids_test, attention_mask=attention_mask_test)

        # Saving the 'features' for logistics regression model
        features_test = last_hidden_states_test[0][:,0,:].cpu().numpy()

        # Saving the 'polarity' of the sentence to the 'labels'
        labels_test = df_test['polarity']

        # Loading and Predicting the model
        filename = 'final_model.h5'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(features_test)
        return result

