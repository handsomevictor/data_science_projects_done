Class Project for a Kaggle Competition.

Course Name: Machine Learning in Network Science

https://www.kaggle.com/competitions/cs-mlns-22



# Introduction

In this project, we predicted missing links in a citation network using graph structure and node information to build features and multiple classifier models.



## Feature engineering

In order to represent the edges we intend to classify, we introduced a set of features relying on the graph structure as well as the node information provided. Note that we tried multiple models at the same time, so we tested our results by batches of submissions. Some of the features only appeared in the later batches (mostly in the third batch).



## Graph features

The first set of features which are relying on the graph structure are the following:



**source\_degree\_centrality**: Degree centrality of the source node;
**target\_degree\_centrality**: Degree centrality of the target node;
**pref\_attach**: Preferential attachment score of the two nodes;
**aai**: Adamic Adar index of the two nodes;
**jacard\_coeff**: Jaccard coefficient of the two nodes;
**diff\_bt**: Difference between the betweenness centralities of the two nodes (this features requires long computation so it was only used in the second batch of submissions);
**common\_neigh**: Number of common neighbors between the two nodes (introduced in the third batch of submissions).



In the centrality features, we try to find which nodes are most likely to be source or target of many other nodes. While in the preferential attachment, Adamic Adar index, Jaccard coefficient and number of common neighbors, we try to use the neighbors of two nodes to estimate the likelihood of a link between the two nodes.



Then, to use the information provided on the nodes (paper title, abstract, publication year and authors), we introduced the following features:



## Node features

**overlap\_title**: Number of common words in the paper titles;
**temp\_diff**: Difference in publication years of the two papers;
**comm\_auth**: Number of common authors in the papers;
**same\_journal\_name**: Boolean indicating if the papers where published in the same journal.



These features are quite simple and try to capture the similarity in sources and subjects of the article. However, the date difference might be he most useful to determine whether a node is source or target of a citation.

However, to further improve on the classification accuracy, we vectorized the abstract using TF-IDF score on the words in the abstract. This way, we intend to use the most common (or frequent) words in the article's abstract to grasp an idea of the subject and find nodes with similar subjects. Thus, we added the following features:



**comm\_top\_words**: Number of common words within the words with the 10 highest TF-IDF score (in the abstract);
**abstract\_sim**: Cosine similarity between the TF-IDF vetors of the papers' abstracts (introduced in the third batch of submissions).



Finally, in the third batch of submissions, to go further on the analysis of the article's subject, we used Topic Modeling on the title and on the abstract to build two last features:

**comm\_title\_topics**: Boolean indicating if the main topic detected in the paper titles is the same (among 5 topics);
**comm\_abstract\_topics**: Number of common topics detected in the abstracts (among 10 topics).



# Classifier models

In order to learn the classification of edges, we relied on 9 classifier models available in the Scikit-Learn library. We compared these models against each other to find the one that best fitted our problem. The models are the following:

Nearest Neighbors models, with k=3 neighbors;
Linear SVM and Radial Basis Function (RBF) kernel SVM;
Decision Tree, with a maximum depth of 10;
Random Forest, with a maximum depth of 10 and 100 estimators;
Neural Network, with one hidden layer at first, and then two and three hidden layers;
AdaBoost, Gaussian Naive Bayes and Quadratic Discriminant Analysis.



# Results

## Model comparison

We ran a first batch of submissions with all models on 5% of the available training data and the first 10 features we had built. This way we could quickly identify the most promising models in our use case. The results of this models are displayed in the table **first_batch**. We could already identify that the best performing models would be the KNN, the RBF SVM, the Neural Network and the AdaBoost Classifier (we will also consider the Random Forest Classifier since it usually performs nicely on these problem). Thus we focused our next studies on these classifier models.

| Classifier model  | Test F1-Score |
| ----------------- | ------------- |
| Nearest Neighbors | 0.95497       |
| Linear SVM        | 0.95123       |
| RBF SVM           | **0.95950**   |
| Decision Tree     | 0.93841       |
| Random Forest     | 0.93517       |
| Neural Net        | **0.95499**   |
| AdaBoost          | 0.93997       |
| Naive Bayes       | 0.92907       |
| QDA               | 0.80821       |

**F1-Scores on the first batch of submissions, with 10 features and 5% of the training data used to fit the classifiers**



Then, in the second batch of submissions, we simply increased the size of the fitting set to 20% of the training data (we also added a graph feature which we dropped later). In the second batch we also added a layer to the Neural Network model which then outperformed all other models. And in the third batch of submissions, we added 4 features which led to the results presented in table second_batch.

| Classifier model           | Train F1-Score      | Test F1-Score  |
| -------------------------- | ------------------- | -------------- |
| Nearest Neighbors          | 0.97571             | 0.96243        |
| RBF SVM                    | Prediction time out | 0.96824        |
| Random Forest              | 0.97609             | 0.97441        |
| Neural Net 2 hidden layers | 0.97513             | 0.97430        |
| Neural Net 3 hidden layers | 0.97500             | extbf{0.97535} |
| AdaBoost                   | 0.97092             | 0.97127        |

**F1-Scores on the fourth batch of submissions, with 14 features and 50\% of the training data used to fit the classifiers**



## Feature impact

Between the second and third batch, we added more topic based features and removed a centrality feature. We observed that for models such as KNN or the RBF SVM, these features had low impact on the test F1-score. Meanwhile, it showed good improvements for the Neural Networks and the AdaBoost classifier. Thus, these more complex models could find patterns in the data that the lower level models couldn't find. We can conclude that in this case the centrality measures might be less important than the topic based features to predict new links.



## Parameter tuning and overfitting

Due to the tight deadline, we focused out parameter tuning on the best performing model, which is Neural Networks. We progressively increased the complexity of the model by adding layers and went from a score of 0.96132 with one 100 nodes layer to 0.97430 with two 100 nodes layers and finally to 0.97535 with three layers.

As for overfitting, to control it we computed the F1-Score on the training set as shown in table **second_batch**. We can see that the more complex models such as the Neural Networks and the AdaBoost classifier do not overfit as much as the simpler models. We can then use their training F1-Score to tune the parameters in the models.

We conducted this parameter tuning on the layer sizes of the Neural Network with 3 hidden layers and found our best results with the layer sizes (100, 100, 20) (through a simple grid search). However, the results on the test set ended up similar throughout all the parameter sets so we kept the best score we reached which was for layer sizes (50, 50, 20). To find the details of the training F1-Scores, refer to the table at the end of the Jupyter Notebook.



# Conclusion

To conclude, our best model reached a F1-Score of 0.97535 on the test set in the Kaggle Competition using a Neural Network with three hidden layers of 50, 50 and 20 nodes respectively trained on 20\% of the training data with 14 features to represent the edges of the graph.
