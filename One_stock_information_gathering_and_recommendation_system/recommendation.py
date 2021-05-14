# -*- coding: utf-8 -*-
from pandas_datareader import data as pdr
from get_all_tickers import get_tickers as gt
import yfinance as yf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# this function helps me to get the ticker names of the largest n market cap companies
top_30 = gt.get_biggest_n_tickers(30)

# create a list that contains all the market cap of the selected tickers as comparisons
all_information_needed = pd.DataFrame()

# this funtion will return a DataFrame which will contain some features of one specific stock
def market_cap_list_function(stock_name, i):
    global all_information_needed
    market_cap_list = 0
    beta = 0
    isEsgPopulated = 0
    quoteType = 0
    enterpriseToRevenue = 0
    Fifty_two_WeekChange = 0
    trailingEps = 0
    priceToBook = 0
    
    all_information = yf.Ticker(stock_name)
    
    market_cap_list = all_information.info['marketCap']
    beta = all_information.info['beta']
    isEsgPopulated = all_information.info['isEsgPopulated']
    quoteType = all_information.info['quoteType']
    enterpriseToRevenue = all_information.info['enterpriseToRevenue']
    Fifty_two_WeekChange = all_information.info['52WeekChange']
    trailingEps = all_information.info['trailingEps']
    priceToBook = all_information.info['priceToBook']
   
    all_information_needed.at[i, 'market_cap_list'] = market_cap_list
    all_information_needed.at[i, 'beta'] = beta
    all_information_needed.at[i, 'isEsgPopulated'] = isEsgPopulated
    all_information_needed.at[i, 'quoteType'] = quoteType
    all_information_needed.at[i, 'enterpriseToRevenue'] = enterpriseToRevenue
    all_information_needed.at[i, 'Fifty_two_WeekChange'] = Fifty_two_WeekChange
    all_information_needed.at[i, 'trailingEps'] = trailingEps
    all_information_needed.at[i, 'priceToBook'] = priceToBook
    return all_information_needed

# now all_information_needed contains all the information we want as training dataset
for q in range(len(top_30)):
    market_cap_list_function(top_30[q], q)

# create a list that contains 10 things: price fluctuation n days ago
creating_list_for_convenience = []
for i in range(1,11):
    a = 'price fluctuation '
    b = i
    c = ' days ago'
    creating_list_for_convenience.append(a+str(b)+c)
creating_list_for_convenience = creating_list_for_convenience[::-1]

# point out other important features
other_features = ['beta', 'isEsgPopulated',
                  'quoteType', 'enterpriseToRevenue',
                  'Fifty_two_WeekChange', 'trailingEps',
                  'priceToBook']
parameters = ['Ticker', 'marketcap'] + other_features + creating_list_for_convenience

# create the final training dataset DataFrame
ML_database = pd.DataFrame(columns = parameters)

# add the ticker to the ML_database first column
for i in range(len(top_30)):
    ML_database.at[i, 'Ticker'] = top_30[i]
    
# add the ticker's marketcap to the ML_database second column
for_all_information = all_information_needed
for i in range(len(top_30)):
    ML_database.at[i, 'marketcap'] = list(for_all_information['market_cap_list'])[i]
    ML_database.at[i, 'beta'] = list(for_all_information['beta'])[i]
    ML_database.at[i, 'isEsgPopulated'] = list(for_all_information['isEsgPopulated'])[i]
    ML_database.at[i, 'quoteType'] = list(for_all_information['quoteType'])[i]
    ML_database.at[i, 'enterpriseToRevenue'] = list(for_all_information['enterpriseToRevenue'])[i]
    ML_database.at[i, 'Fifty_two_WeekChange'] = list(for_all_information['Fifty_two_WeekChange'])[i]
    ML_database.at[i, 'trailingEps'] = list(for_all_information['trailingEps'])[i]
    ML_database.at[i, 'priceToBook'] = list(for_all_information['priceToBook'])[i]

# add all the price of the last ten days
for i in range(len(top_30)):
    processed = pdr.get_data_yahoo(top_30[i], "2020-11-10").iloc[-11:, 5]
    processed = pd.DataFrame(processed)
    processed['fluctuation'] = 0
    for m in range(10):
        processed.iloc[m+1, 1] = processed.iloc[m+1, 0]/processed.iloc[m, 0] - 1
    processed = processed.iloc[1:, :]
    
    processed_ticker_list_number = list(ML_database['Ticker']).index(top_30[i])
    for n in range(10):
        ML_database.iloc[processed_ticker_list_number, n+9] = processed.iloc[n, 1]

# find if there are nulls in the DataFrame, if there are, replace them with 0
ML_database = ML_database.fillna(0)

# replace all False with 0, all True with 1, all EQUITY kinds with 1, all other kinds with 0
for i in range(len(top_30)):
    if ML_database.at[i,'isEsgPopulated'] == 0:
        ML_database.at[i,'isEsgPopulated'] = 1
        ML_database['isEsgPopulated'] = ML_database['isEsgPopulated'].astype(int)
    else:
        ML_database.at[i,'isEsgPopulated'] = 0
    if ML_database.at[i,'quoteType'] == 'EQUITY':
        ML_database.at[i,'quoteType'] = 1
    else:
        ML_database.at[i,'quoteType'] = 0

# add label to the training data we have, and the label means if we should buy it or not
# Whether the stock is worth buying or not depends on the equity researchers' report
ML_database['label'] = 0

# this label is derived from financial institutions' equity research reports
# If the ticker is "strongly recommended", then I give it a 1 (which means buy),
# if it's not strongly recommended, then label it "not recommended", which is sell
label_from_equity_research = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]

for i in range(len(top_30)):
    ML_database.at[i, 'label'] = label_from_equity_research[i]
    
# now we have all the training data
# since the stock pitching problem should be more reserved and risk-averse, we should
# not be over-fitting at all, and the model should be strong in noise cancellation.
# Also, the data we have now is not very balanced. Therefore, we decided to adopt a random
# forest model to train the dataset.
total_train = ML_database
# delete all non-number things, because we have enough numerical features
total_train = total_train.iloc[:,1:]

# write in the report why I chose 0.7 this parameter! And why I chose the following parameters!
X_train, X_test, Y_train, Y_test = train_test_split(total_train.iloc[:, :18], total_train.label, train_size = 0.8)

param_grid = {'criterion': ['entropy'],
              'max_depth': [3],
              'n_estimators': [9],
              'max_features': [0.4],
              'min_samples_split': [2]}

rfc = ensemble.RandomForestClassifier()

rfc_cv = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 4)

rfc_cv.fit(X_train, Y_train)

predict_test = rfc_cv.predict(X_test)

# get the f1 score report
matrix_in_string = metrics.classification_report(predict_test, Y_test)

# this f1_score is derived from the metrix table, because the table output type is str, so
# I used the slicing to get the f1 score
f1_score_average = float(matrix_in_string[272+39:272+43])
accuracy = float(matrix_in_string[203:207])

#----------------------------------------------------------------------------
# from running the GridSearchCV for 20 times, we got an average parameter which is 
# the one I used above, and now it's time to give the investor's stock a rating of 
# buy or not buy.

# The following function is to make target stock into the same DataFrame as the one we did for training data,
# so that it will make the process of getting the outcome on the test dataset faster.
def machine_learning_to_recommend_buy_or_not(ticker):
    all_information_for_test_needed = market_cap_list_function(ticker, 0)
    all_information_for_test_needed = all_information_for_test_needed.iloc[0,:]
    # create a list that contains 10 things: price fluctuation n days ago
    creating_list_for_test_convenience = []
    for i in range(1,11):
        a = 'price fluctuation '
        b = i
        c = ' days ago'
        creating_list_for_test_convenience.append(a+str(b)+c)
    creating_list_for_test_convenience = creating_list_for_test_convenience[::-1]
    
    # for convenience
    global ML_test_database
    ML_test_database = pd.DataFrame(columns = parameters)
    i = 0
    ML_test_database.at[i, 'Ticker'] = ticker
    ML_test_database.at[i, 'marketcap'] = list(for_all_information['market_cap_list'])[i]
    ML_test_database.at[i, 'beta'] = list(for_all_information['beta'])[i]
    ML_test_database.at[i, 'isEsgPopulated'] = list(for_all_information['isEsgPopulated'])[i]
    ML_test_database.at[i, 'quoteType'] = list(for_all_information['quoteType'])[i]
    ML_test_database.at[i, 'enterpriseToRevenue'] = list(for_all_information['enterpriseToRevenue'])[i]
    ML_test_database.at[i, 'Fifty_two_WeekChange'] = list(for_all_information['Fifty_two_WeekChange'])[i]
    ML_test_database.at[i, 'trailingEps'] = list(for_all_information['trailingEps'])[i]
    ML_test_database.at[i, 'priceToBook'] = list(for_all_information['priceToBook'])[i]
    
    # add all the price of the last ten days
    processed_test = pdr.get_data_yahoo(ticker, "2020-11-10").iloc[-11:, 5]
    processed_test = pd.DataFrame(processed_test)
    processed_test['fluctuation'] = 0
    for m in range(10):
        processed_test.iloc[m+1, 1] = processed_test.iloc[m+1, 0]/processed_test.iloc[m, 0] - 1
    processed_test = processed_test.iloc[1:, :]
    
    processed_test_ticker_list_number = list(ML_test_database['Ticker']).index(ticker)
    for n in range(10):
        ML_test_database.iloc[processed_test_ticker_list_number, n+9] = processed_test.iloc[n, 1]
    
    # find if there are nulls in the DataFrame, if there are, replace them with 0
    ML_test_database = ML_test_database.fillna(0)
    
    # change every non number things to numbers
    i = 0
    if ML_test_database.at[i,'isEsgPopulated'] == 0:
        ML_test_database.at[i,'isEsgPopulated'] = 1
        ML_test_database['isEsgPopulated'] = ML_test_database['isEsgPopulated'].astype(int)
    else:
        ML_test_database.at[i,'isEsgPopulated'] = 0
    if ML_test_database.at[i,'quoteType'] == 'EQUITY':
        ML_test_database.at[i,'quoteType'] = 1
    else:
        ML_test_database.at[i,'quoteType'] = 0
    
    # now use the model to predict if this stock should be bought or not: 1 is to buy
    ML_test_database = ML_test_database.iloc[:,1:]
    global predict_real_test
    predict_real_test = rfc_cv.predict(ML_test_database)
    predict_real_test = int(predict_real_test)
    
    # change the output to normal language
    if predict_real_test == 1:
        predict_real_test = 'Go and buy it now! Accuracy=' + str(accuracy)
    elif predict_real_test == 0:
        predict_real_test = "AI think it's not a good idea to buy it now! Accuracy=" + str(accuracy)
    return predict_real_test


