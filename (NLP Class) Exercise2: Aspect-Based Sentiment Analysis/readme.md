The goal of this exercise is to implement a classifier to predict aspect-based polarities of opinions in sentences.

Name of the students:
Zhenning Li
Nevina Dalal
Maria Isabel Vera Cabrera
Konstantinos Mira


Introduction:
Sentiment analysis is one of the classical tasks concerning Natural Language Processing and it has gained increasing popularity among research and 
business due to the ever increasing amount of opinionated text from internet users. Standard sentiment analysis involves classifying the overall 
sentiment of an entire instance of text. However, a piece of text rarely carries only one sentiment. Moreover, we may be interested in the topics 
or aspects of different parts of the text and what those individual sentiments may be. For instance, a review from an airline passenger may read: 
“The service was great, but the food was awful!” The airline may be interested in learning about both, the sentiments of their service and of 
their catering to make their business decisions. Thus, in this project we examine an approach to Aspect-based sentiment analysis (ABSA). 
ABSA is a more complex task that consists in identifying both sentiments and aspects. This report aims to show the potential of using the contextual 
word representations from the pre-trained language model DistilBERT and a Classifier. We also ran multiple experiments on the pre-processing of the 
text, baseline standard sentiment analysis, and different supplemental Classifiers for the DistilBERT approach.


Pre-processing (Initial approach):
Although the DistilBERT + Classifier method does not require text preprocessing before hand, for the baseline model of the TF IDF + Classifier method, 
the text was treated so that an instance of a text was subsetted to the particular aspect, in a supervised learning environment way. For our example 
earlier, two different instances in the dataset could be that same sentence, “The service was great, but the food was awful!” , but have different 
‘target_term’s or aspects, as well as aspect_category , as two different rows in the dataset. For instance, one with target_term as ‘service’ and the 
other as ‘food’. To do so, First we prepared the original text by removing alphanumeric characters, urls, and lowercasing. Then by splitting text into 
sentences, then words, performing POS tagging, and removing stop words. After this, the text object is now ready to be fed into the pipeline which will 
use a target_term and aspect_category combination and the pre-processed text object to subset the original text into the relevant portion of the text. 
This output was labeled as opinion_words. As the class distribution was imbalanced, over 70% of the dataset is labeled as positive, we also tried to 
augment the opinion_words text by performing simple unsampling. After vectoring the opinion_words instances using a TF IDF vectoriser, the performance 
of a OneVsRestClassifier with Logistic Regression on a validation set was impressive with an accuracy score of 0.77 (given that the train and validation 
split was 0.95 / 0.05, respectively). However, the performance of this simple model on the development set (after being pre-processed as well) was only 
0.60 accuracy score. Thus, this motivated the need for a Bidirectional Encoder Representations from Transformers (BERT) model, which is a 
transformer-based machine learning technique for natural language processing pre-training developed by Google.


Pre-trained model - DistilBERT:
DistilBERT processes the sentence and passes along some information it extracted from it onto the next model. DistilBERT is a smaller version of BERT 
developed and open sourced by the team at HuggingFace. It's a lighter and faster version of BERT that roughly matches its performance.

Methodology:
1. Load the Pre-trained BERT model.
2. Preparing the Dataset.
a) Tokenization: Tokenize the sentences -- break them up into word and sub words in the format BERT is comfortable with.
b) Padding: After tokenization, tokenized is a list of sentences -- each sentence is represented as a list of tokens. 
BERT processes our examples all at once (as one batch). 
It's just faster that way. For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-d array, 
rather than a list of lists (of different lengths).
c) Masking: If we directly send padded to BERT, that would slightly confuse it. 
We need to create another variable to tell it to ignore (mask) the padding we've added when it's processing its input. 
3. Training the model using Deep Learning. The Pre-trained Bert model function runs our sentences . 
The results of the processing will be returned.
4. We sliced only the part of the output that we needed. That is the output corresponding to the first token of each sentence. 
The way BERT does sentence classification is that it adds a token called [CLS] (for classification) at the beginning of every sentence. 
The output corresponding to that token can be thought of as an embedding for the entire sentence. 
5. We'll save it in the features variable. It serves as the features to our classifier model and labels are the polarity in the data. 
The training is done with a classifier and the weights of the model are saved. 
6. The same series of above steps are used for the devdata. Then the model is loaded with weights saved and predicts the results.


Classifier Model:
The classifier will take in the result of DistilBERT's processing, and classify the sentence as either positive, negative or neutral. We tried 
different classification models such as Logistic Regression, Support Vector Machines, Random Forest and One vs Rest classifier with logistic 
regression. Those models obtained an initial accuracy on the development set of 81.15, 70.21, 79.36 and 80.85 respectively. 
Noticing that SVM got the lowest score and that the OneVsRest method decreased the original model’s performance, we decided to focus on the two 
remaining models. For Random forest, we doubled the number of estimators from 64 to 128 trees and although the metric slightly rose to 80.16, it 
was still lower than the one obtained by LR while the computational time was significantly increased.
We decided then to focus on improving our best classifier so far: Logistic Regression. For this purpose, we used GridSearch cross validation for 
hyperparameter tuning. In our parameter grid we included 20 evenly spaced values for penalty strength ( C ) in a range from 0.0001 to 100. 
For the algorithm to use in the optimization problem we tried newton-cg, lbfgs and liblinear, all with a L2 norm penalty. To evaluate all 
combinations, we used Repeated Stratified 10-Fold which applied different randomization in each of the 3 repetitions. The best cross validation 
results on the train set were obtained with C=5.25 and solver=liblinear, for a result of 82.83. Although the grid search technique optimizes the 
cv performance on the train set, it does not show a result as high when evaluated on the development set. After trying different combinations of 
parameters, the chosen ones were a higher penalty of C=0.1 (lower values specify stronger regularization), L2 penalty term, lbfgs solver as the 
optimization method and 2000 as the maximum number of iterations to converge. 
With this final configuration, we reached the highest accuracy on the development dataset equal to 82.45.
