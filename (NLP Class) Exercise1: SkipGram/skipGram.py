#import libraries
from __future__ import division
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import normalize
import pickle

__authors__ = ['Zhenning Li',
			   'Konstantinos Mira',
			   'Maria Isabel Vera Cabrera',
			   'Nevina Dalal']

__emails__  = ['zhenning.li@student-cs.fr',
			   'konstantinos.mira@student-cs.fr',
			   'maria-isabel.vera-cabrera@student-cs.fr',
			   'nevina.dalal@student-cs.fr']

def text2sentences(path):
	"""
	This function is for preprocessing all the data.
	1. replace every special characters with empty '' except "'" because maybe something like "I'd like to do NLP homework" may exist
	2. delete all numbers
	3. delete all blanks
	"""

	# first determine the special characters, the following list is copied from here:
	# https://www.codegrepper.com/code-examples/python/all+special+characters+python
	special_list = {'~', ':', '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}',
					'.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/'}
	# add escape characters
	escape_list = {'\\', '\\n', '\\r', '\\t', '\\b', '\\f', '\\ooo', '\\xhh'}
	escape_list = list(special_list.union(escape_list))

	sentences_text2sentences = []

	#with train.txt file
	with open(path) as f:

		#for each line (sentence) in the text
		for l in f: 

			# replace special characters
			for special in escape_list:
				l = l.replace(special, '')

			# delete numbers
			for i in range(10):
				l = l.replace(str(i), '')

			# split the sentence into words separated by commas and save them in a list
			# convert all text to lowercase
			# only include words with more than 1 character
			# remove spaces at the beginning and at the end of the word
			l = [final.strip() for final in l.lower().split() if len(final.strip())>1]

			# append pre processed line (list of words)
			sentences_text2sentences.append(l)
	 
	#return list of lists with cleaned words (one list per sentence)
	return sentences_text2sentences

def loadPairs(path):
		#simlex.csv
    data = pd.read_csv(path, delimiter='\t')
		#creates tuples (word1, word2, similarity) for each row (word pairs) in the data
    return zip(data['word1'], data['word2'], data['similarity'])

def sigmoid(x):
	#logistic function used in the loss/objective function
	return 1/(1+np.exp(-x))

class SkipGram:

    def __init__(self, sentences, nEmbed=300, negativeRate=5, winSize = 5, minCount = 5):
        
        # Hyperparameters
        self.nEmbed = nEmbed #dimensionality of embeddings
        self.negativeRate = negativeRate #number of negative samples (incorrect training pair instances) that are drawn for each good sample
        self.winSize = winSize #how many words to the left and to the right are considered contexts of the target
        self.minCount = minCount #min frequency for a word to be included in the vocabulary
        self.lr = 0.001 #learning rate
        
        self.sentences=sentences
        
        # Dictionary to store the frequency of each word
        self.word_counts = {}

        # Initializing dictionary
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] = 0
				
        # Counting words
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1

        # Add 'unknown' word to dictionary in case of new data
        self.word_counts['unknown'] = 0

        # Filter to include only words that have a frequency >= minCount
        if self.minCount > 1:
            self.word_counts = {word:freq for word, freq in self.word_counts.items() if freq >= self.minCount or word=='unknown'}

        # Number of unique words
        self.w_count = len(self.word_counts.keys())
        
        # List of unique words
        self.vocab = list(self.word_counts.keys())

        # Dictionary with word to ID mapping
        self.w2id = dict((word, i) for i, word in enumerate(self.vocab)) 

        # Dictionary with ID to word mapping
        self.id2w = dict((i, word) for i, word in enumerate(self.vocab)) 
        
        #Initialize unigram array with zeros
        self.unigram_table = np.zeros(sum(self.word_counts.values()), dtype=np.uint32)

        i_freq = 0 
        power=0.75
        for word, freq in self.word_counts.items():
            #Frequency for each word raised to desired power and converted to integer
            freq = np.uint32((np.ceil(freq ** power)))
            for _ in range(freq):
                #Update unigram array where each word id appears as many times as its new calculated frequency
                self.unigram_table[i_freq] = self.w2id[word]
                i_freq += 1

        #Update unigram table size with the sum of all frequencies
        self.unigram_size = i_freq
        self.unigram_table = self.unigram_table[:i_freq]
        
        # Initialize random matrix W: embedding, with the number of unique words as rows and the dimension of embeddings as columns
        self.W = np.random.randn(self.w_count, nEmbed)

        #Initialize random matrix C: context, with the number of unique words as rows and the dimension of embeddings as columns
        self.C = np.random.randn(self.w_count, nEmbed)

    def sample(self, omit):
        
        # Convert set of words to omit to list
        omit = list(omit)

        # Get the id of the words to omit (target and context word)
        widx=self.id2w.get(omit[0])
        cidx=self.id2w.get(omit[1])
        
        # Get random samples from the unigram table. The number of samples is equal to the parameter negativeRate
        samples = self.unigram_table[np.random.randint(0, high=self.unigram_size, size = self.negativeRate)]
        
        for i, sampled_idx in enumerate(samples):
        #If the sample is equal to the any of the words to omit, sample again and replace that word
            while sampled_idx == cidx or sampled_idx == widx:
                sampled_idx = self.unigram_table[np.random.randint(0, high=self.unigram_size, size = self.negativeRate)]
                samples[i] = sampled_idx

        #Return the indexes of the negatively sampled words
        return samples
        
    def train(self):

        #Initialize variables
        self.trainWords = 0
        self.accLoss= 0
        self.loss = []
        self.loss_rec_ = []
        
        for counter, sentence in enumerate(self.sentences):
            #Include only words that exist in the pre-defined vocabulary
            sentence = list(filter(lambda word: word in self.vocab, sentence))

            for wpos, word in enumerate(sentence):
                #Id for the word
                wIdx = self.w2id[word]
                #Define a window size as a random number between 1 and the initial window size+1
                winsize = np.random.randint(self.winSize) + 1
                #Define index position in the sentence for the start of the window
                start = max(0, wpos - winsize)
                #Define index position in the sentence for the end of the window
                end = min(wpos + winsize + 1, len(sentence))

                #For each word in the window (context)
                for context_word in sentence[start:end]:
                #Extract the id of the context word
                    ctxtId = self.w2id[context_word]
                #If the context word is equal to the target word, omit the following instructions and go to next word
                    if ctxtId == wIdx: continue
                #Get ids of negative sampled words
                    negativeIds = self.sample({wIdx, ctxtId})
                #Update W and C matrices
                    self.trainWord(wIdx, ctxtId, negativeIds)
                #Increase the count of trained words
                    self.trainWords += 1
                #Add loss from current word
                    self.accLoss += self.loss_function(wIdx, ctxtId, negativeIds)

            if counter % 100 == 0:
                #Every 100 sentences, print status
                print(' > training %d of %d' % (counter, len(self.sentences)))
                #save Average loss in current batch
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0
                #Print average loss for the current 100 sentences
                print(self.loss[-1])
                self.loss_rec_.append(self.loss[-1])
                #return self.loss_rec_

    def loss_function(self, wordId, contextId, negativeIds):
        #The objective of the model is to maximise the probability of the correct samples coming from the corpus 
        #and minimise the corpus probability for the negative samples
        l_sum = sigmoid(np.dot( -self.W[wordId, :], self.C[contextId, :] ))
        for negativeId in negativeIds:
            l_sum *= 1 - sigmoid(np.dot( -self.W[negativeId, :], self.C[contextId, :]))
        return l_sum
                       
    def trainWord(self, wordId, contextId, negativeIds):
                        
        W_update = 0 
        #Calculate W' matrix
        W_update -= (sigmoid(np.dot(self.W[wordId,:], self.C[contextId, :])) - 1) * self.C[contextId, :]
        
        #For each negatively sampled word
        for negativeId in negativeIds:
            #Update C context matrix
            self.C[contextId, :] -= self.lr * sigmoid(np.dot(self.W[negativeId,:], self.C[contextId, :])) * self.W[wordId, :]
            #Update W'  matrix
            W_update -= sigmoid(np.dot(self.W[negativeId,:], self.C[contextId, :])) * self.C[contextId, :]
        
        #Update W taking into account the learning rate parameter
        self.W[wordId, :] -= self.lr * W_update
        
    def save(self, path):
        """
        save the model to .pickle
        """
        with open(path, 'wb') as f:
            #save self (skipGram instance)
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)       

    def similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        
        # If the two words are equal, similarity is 1
        if word1 == word2:
            print("same word")
            return 1
        
        # Checks if word is in vocabulary
        if word1 in self.w2id:
            id1 = self.w2id[word1]
        else:
        #if not it is 'unknown'
            id1 = self.w2id['unknown']

        # Checks if word is in vocabulary		
        if word2 in self.w2id:
            id2 = self.w2id[word2]
        else:
            #if not it is 'unknown'
            id2 = self.w2id['unknown']

        #Calculate cosine similarity score between the two words
        cos_sim = np.dot(self.W[id1,:], self.W[id2,:])/(np.linalg.norm(self.W[id1,:])*np.linalg.norm(self.W[id2,:]))
        
        #The cosine similarity always belongs to the interval [-1,1]
        if cos_sim > 1:
            cos_sim = 1
        elif cos_sim < -1:
            cos_sim = -1

        #return (cos_sim + 1)/2
        return cos_sim 

    @staticmethod
    def load(path):
        """
        load the model for doing test
        """

        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
       

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)

        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
