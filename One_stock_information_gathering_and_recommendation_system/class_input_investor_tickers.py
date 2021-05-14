# -*- coding: utf-8 -*-
import pandas as pd
#import numpy as np
import datetime
#import matplotlib.pyplot as plt
#from datetime import timedelta
from pandas_datareader import data as pdr
#import fix_yahoo_finance as yf
#import openpyxl
import calendar

#-----------------------------------------------------------------------------
# This file, is to make investor's input information into a DataFrame, with some basic methods to judge 
# if the things investor just input is legal.

# initialization of DataFrame
def dataframe_initialization():
    global data_base
    parameters = ['Acquisition Date', 'Ticker', 'Quantity', 'Unit Cost', 'Cost Basis','Start of Year' ]
    data_base = pd.DataFrame(columns = parameters)
    return data_base
#------------------------------------------------------------------
# Initialization of the things that investors are ordered to input
def input_process():
    
    global ticker
    ticker = 0
    global ticker_acquisition_date
    ticker_acquisition_date = 0
    global ticker_quantity
    ticker_quantity = 0
    global ticker_unit_cost
    ticker_unit_cost = 0
    global ticker_cost_basis
    ticker_cost_basis = 0
    global ticker_start_of_year
    ticker_start_of_year = 0
    
# basic error information input validation process
    ticker = input('please input your ticker')
    ticker = ticker.upper()
    ticker_acquisition_date = input('please input your acquisition date in the form: "Year-Month-Day"')
    ticker_quantity = input('please input your quantity')
    ticker_unit_cost = input('please input your unit cost')
    
    ticker_cost_basis = float(ticker_quantity)* float(ticker_unit_cost)
    
# we will use the price at the end of last year to make some graphs
    ticker_start_of_year = datetime.datetime(2019,12,31)

#------------------------------------------------------------------
# make sure the ticker the investor input is correct (can be found in the database)
def judge_whether_ticker_is_legal(ticker):
    today = datetime.date.today()
 
    try:
        pdr.get_data_yahoo(ticker, "2020-11-27")
        return True
    except:
        return False

# judge if the number he input is legal (can be recognized in datetime.datetime)
def judge_whether_ticker_acquisition_is_legal_in_format(ticker_acquisition_date):
    ticker_acquisition_date = ticker_acquisition_date.replace('-',' ')
    ticker_acquisition_date = ticker_acquisition_date.split()
    k = []
    for i in ticker_acquisition_date:
        k.append(int(i))
    try:
        datetime.datetime(k[0], k[1], k[2])
        return True
    except:
        return False

# judge if the input date is in weekday (there is no price for weekend since the market is not open)
def judge_whether_ticker_acquisition_is_legal_in_weekdays(ticker_acquisition_date):
    if judge_whether_ticker_acquisition_is_legal_in_format(ticker_acquisition_date) == True:     
        ticker_acquisition_date = ticker_acquisition_date.replace('-',' ')
        ticker_acquisition_date = ticker_acquisition_date.split()
        k = []
        for i in ticker_acquisition_date:
            k.append(int(i))
        if calendar.weekday(k[0], k[1], k[2]) <= 4:
            return True
        else:
            return False
    else: 
        return False

# judge if investors' quantity input is legal (non negative and integer)
def judge_whether_ticker_quantity_is_legal(ticker_quantity):
    try:
        if (float(ticker_quantity) - int((float(ticker_quantity)))) != 0:
            return False
        else:
            return True
    except:
        return False

# judge if investors' unit cost input is legal (any number below 0 or string will not be accepted)
def judge_whether_ticker_unit_cost_is_legal(ticker_unit_cost):
    try:
        if (float(ticker_unit_cost) - int((float(ticker_unit_cost)))) != 0 and float(ticker_unit_cost) <= 0:
            return False
        else:
            return True
    except:
        return False
# change the acquisition date format for the convenience of merging dataframes in another file
def change_acquisition_date_format(ticker_acquisition_date):
    ticker_acquisition_date = ticker_acquisition_date.replace('-',' ')
    ticker_acquisition_date = ticker_acquisition_date.split()    
    k = []
    for i in ticker_acquisition_date:
        k.append(int(i))
    return datetime.datetime(k[0], k[1], k[2])

#------------------------------------------------------------------
# main1() is the function that will make investor input the information of the investments
# that he had made, and generate the information dataframe
ii = 1
def main1():
    global ii
    global ticker_acquisition_date
    
    input_process()
    print(ticker)
    print(ticker_acquisition_date)
    check1 = judge_whether_ticker_is_legal(ticker)
    check2 = judge_whether_ticker_acquisition_is_legal_in_format(ticker_acquisition_date)
    check3 = judge_whether_ticker_acquisition_is_legal_in_weekdays(ticker_acquisition_date)
    check4 = judge_whether_ticker_quantity_is_legal(ticker_quantity)
    check5 = judge_whether_ticker_unit_cost_is_legal(ticker_unit_cost)
    
    # if there is no error in the inputs, proceed to the next step
    if (check1 and check2 and check3 and check4 and check5):
        
        ticker_acquisition_date = change_acquisition_date_format(ticker_acquisition_date)
        
        data = {}
        data['Acquisition Date'] = ticker_acquisition_date
        data['Ticker'] = ticker
        data['Quantity'] = ticker_quantity
        data['Unit Cost'] = ticker_unit_cost
        data['Cost Basis'] = ticker_cost_basis
        data['Start of Year'] = ticker_start_of_year
        
        data_base.loc[ii] = data
        
        # we allow investors to input as many as he wants
        want_continue = input('Do you want to add more tickers? y/n')
        if want_continue == 'y':
            ii += 1
            main1()
            return data_base
        
        elif want_continue == 'n':
            # data_base.set_index('Acquisition Date', inplace = True)
            return data_base
        
        else:
            print('input error, try it again')
            main1()
            return None
    # if the flow goes into this step, it means some inputs are not legal. The illegal input will be recognized
    # and sent to investor
    else:
        if not check1:
            print('ERROR! ticker is not found')
        elif not check2:
            print('ERROR! acquisition data is not in correct form')
        elif not check3:
            print('ERROR! the quantity you have input was not correct')
        elif not check4:
            print('ERROR! the unit cost you have input is not correct')
        
        want_continue = input('Do you want to add more tickers? y/n')
        if want_continue == 'y':
            ii += 1
            main1()
            return data_base
        elif want_continue == 'n':
            return data_base
        else:
            print('input error, try it again')
            main1()
            return None        

# We defind this function for investors to get the latest stock price of his input ticker
def get_the_ticker_price(ticker):
    today = datetime.date.today()
    try:
        price_df = pdr.get_data_yahoo(ticker, datetime.datetime(
                                            int(today.strftime('%Y')), 
                                            int(today.strftime('%m')), 
                                            int(today.strftime('%d')),0,0))
        print('The Current Price of ' + ticker + ' is ' + str(price_df['Adj Close'][0]))
    except:
        print('the ticker you have input is wrong! Please check!')