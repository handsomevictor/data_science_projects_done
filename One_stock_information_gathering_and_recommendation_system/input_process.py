# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# import standardized packages
import pandas as pd
import numpy as np
import datetime
import time as time
import calendar
from pandas_datareader import data as pdr
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

#-----------------------------------------------------------------------------
def investor_please_input(data_base):
    
    global portfolio_df
    portfolio_df = data_base
    # make the format of datetime into datestamp, or there will be bugs when merging different dataframs.
    def Changestamp(dt1):
        Unixtime = time.mktime(time.strptime(dt1.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S'))
        return Unixtime

    # reset index to numbers, to add a column for store the datestamp of the date
    portfolio_df = portfolio_df.reset_index()
    
    for i in range(len(portfolio_df)):
        portfolio_df.at[i, 'datestamp'] = Changestamp(portfolio_df['Acquisition Date'][i])
       
    # from now on, when we are doing merging thing, we will always use the datestamp as an indicator

    #-----------------------------------------------------------------------------
    # we need to define yesterday, especially to the format that can be recognized by
    # 'Date' column: datetime.datetime(year, month, day, hour, minute)
    def getYesterday(): 
        today = datetime.date.today()
        oneday = datetime.timedelta(days=1) 
        yesterday = today-oneday  
        
        year = int(yesterday.strftime('%Y'))
        month = int(yesterday.strftime('%m'))
        day = int(yesterday.strftime('%d'))
        return datetime.datetime(year, month, day, 0, 0)
    
    # same reason as getYesterday()
    def getToday():
        today = datetime.date.today()
        year = int(today.strftime('%Y'))
        month = int(today.strftime('%m'))
        day = int(today.strftime('%d'))
        return datetime.datetime(year, month, day, 0, 0)
    
    #-----------------------------------------------------------------------------
    # all stuff about time is here
    # define the earliest time of SP
    global start_sp
    start_sp = datetime.datetime(2005, 1, 1)

    global end_sp
    global today_sp
    year_, month_, day_, hour_, minute_, second_, *_ = datetime.date.today().timetuple()
    if calendar.weekday(year_, month_, day_) <= 4:
        end_sp = getToday()
        today_sp = getToday()
    elif calendar.weekday(year_, month_, day_-1) <= 4:
        end_sp = getYesterday()
        today_sp = getYesterday()
    else:
        end_sp = datetime.datetime(year_, month_, day_ - 2, minute_, second_)
        today_sp = datetime.datetime(year_, month_, day_ - 2, minute_, second_)

    global end_of_last_year
    end_of_last_year = datetime.datetime(2019, 12, 31)
    
    global stocks_start
    stocks_start = datetime.datetime(2005, 1, 1)
    
    global stocks_end_yesterday
    stocks_end_yesterday = getYesterday()
    
    global stocks_end_today
    stocks_end_today = getToday()
    
    #-----------------------------------------------------------------------------
    # This part deals with S&P 500
    # remember! first judge whether the market is open on that day!!!!!!!!! 
    def stocks_time_base(year = 2010, month = 3, day = 9, hour = 0, minute = 0):
        return datetime.datetime(year, month, day, hour, minute)
    
    # get data for S&P 500, as we want to compare the return rate of investor's ticker with S&P 500
    global sp500
    sp500 = pdr.get_data_yahoo('^GSPC', start_sp, today_sp)
    
    # Create a dataframe with only the Adj Close column.
    global sp_500_adj_close
    sp_500_adj_close = sp500[['Adj Close']].reset_index()
    
    # change the date of sp_500_adj_close into data stamp format
    sp_500_adj_close['datestamp2'] = 0
    for i in range(len(sp_500_adj_close)):
        sp_500_adj_close.at[i, 'datestamp2'] = Changestamp(sp_500_adj_close['Date'][i])
    
    global sp_500_adj_close_start
    sp_500_adj_close_start = sp_500_adj_close[sp_500_adj_close['Date'] == end_of_last_year]
    
    global tickers
    tickers = portfolio_df['Ticker'].unique()
    
    # make all chosen stock together in one DataFrame
    def get(tickers, startdate, enddate):
        
        # This function is only used for later map()'s convenience!
        def data(ticker):
            return (pdr.get_data_yahoo(ticker, start = startdate, end = enddate))
        
        datas = map(data, tickers)
        # to merge the stocks we chose
        return(pd.concat(datas, keys = tickers, names = ['Ticker', 'Date']))
    
    #-----------------------------------------------------------------------------
    # This part deals with data manipulation in dataframe after we get the inputs from investors.
    global all_data
    all_data = get(tickers, stocks_start, stocks_end_today)
    
    # We only need 3 columns of the DataFrame all_data
    global adj_close
    adj_close = all_data[['Adj Close']].reset_index()
    
    # get the price of chosen stocks at stock_time_base time point
    global adj_close_start
    adj_close_start = adj_close[adj_close['Date'] == end_of_last_year]
    
    # the latest stock price
    # we cannot use getToday, because the data source sometimes is not that updated
    global adj_close_latest
    
    year1, month1, day1, hour1, minute1, second1, *_ = getYesterday().timetuple()
    if calendar.weekday(year1, month1, day1) <= 4:
        adj_close_latest = adj_close[adj_close['Date'] == getYesterday()]
    elif calendar.weekday(year1, month1, day1-1) <= 4:
        adj_close_latest = adj_close[adj_close['Date'] == datetime.datetime(year1, month1, day1 - 1, minute1, second1)]
    else:
        adj_close_latest = adj_close[adj_close['Date'] == datetime.datetime(year1, month1, day1 - 2, minute1, second1)]
        
    # change the first column to tickers' name, so that we can merge them according to the names
    adj_close_latest.set_index('Ticker', inplace = True)
    portfolio_df.set_index(['Ticker'], inplace = True)
    
    # now we have our cost and the latest price etc, we need to merge them
    global merged_portfolio
    merged_portfolio = pd.merge(portfolio_df, adj_close_latest, left_index = True, right_index = True)
    
    # change the 'Unit Cost' column into float type
    merged_portfolio = merged_portfolio.reset_index()
    merged_portfolio['Unit Cost'] = merged_portfolio['Unit Cost'].astype('float')
        
    # add a column to show the rate of return based on the cost indicated in the investor information
    merged_portfolio['ticker return'] = merged_portfolio['Adj Close'] / merged_portfolio['Unit Cost'] - 1
    
    # reset index, because we have finished merging process, now we want to use the
    # number of index instead of tickers' name
    merged_portfolio.reset_index(inplace = True)
    
    # now we want to merge S&P 500 index and portfolio price based on the time to get the corresponding 
    # S&P 500 index on that stock Acquisition Date.
    global merged_portfolio_sp
    merged_portfolio_sp = pd.merge(merged_portfolio, sp_500_adj_close, left_on = 'datestamp', right_on = 'datestamp2', how = 'left', validate = 'm:1')
    
    # the column 'Date_y' is a replicate of the Acquisition Date
    del merged_portfolio_sp['Date_y']
    
    # change some names to make the DataFrame more understandable
    merged_portfolio_sp.rename(columns = {'Date_x': 'Latest Date', 'Adj Close_x':
          'Ticker Adj Close', 'Adj Close_y': 'SP 500 Initial Close'}, inplace=True)
    
    # this column determines what SP 500 equivalent purchase would have been at the purchase date of stock
    merged_portfolio_sp['Equiv SP Shares'] = merged_portfolio_sp['Cost Basis'] / merged_portfolio_sp['SP 500 Initial Close']
    
    # add a column to show the latest SP 500 Close
    global merged_portfolio_sp_latest
    merged_portfolio_sp_latest = pd.merge(merged_portfolio_sp, sp_500_adj_close, left_on = 'Latest Date', right_on = 'Date')
    
    # delete duplicate column
    del merged_portfolio_sp_latest['Date']
    
    # rename
    merged_portfolio_sp_latest.rename(columns = {'Adj Close': 'SP 500 Latest Close'}, inplace = True)
    
    #-----------------------------------------------------------------------------
    # In this part, we will calculate some ratios for investors to understand his portfolio's performance with S&P 500's performance
    # this column 'SP Return' shows the return rate of SP 500 index from the initial date
    merged_portfolio_sp_latest['SP Return'] = merged_portfolio_sp_latest['SP 500 Latest Close'] / merged_portfolio_sp_latest['SP 500 Initial Close'] - 1
    
    # calculate the difference between SP 500 return and the Ticker return
    merged_portfolio_sp_latest['Absolute Return Compare'] = merged_portfolio_sp_latest['ticker return'] - merged_portfolio_sp_latest['SP Return']
    
    # calculate the total value of each stock investor bought
    merged_portfolio_sp_latest['Quantity'] = merged_portfolio_sp_latest['Quantity'].astype('float')
    
    merged_portfolio_sp_latest['Ticker Share Value'] = merged_portfolio_sp_latest['Quantity'] * merged_portfolio_sp_latest['Ticker Adj Close']
    
    # calculate what the ticker price should be if we use the return rate of SP 500
    merged_portfolio_sp_latest['SP 500 Value'] = merged_portfolio_sp_latest['Equiv SP Shares'] * merged_portfolio_sp_latest['SP 500 Latest Close']
    
    # this is a new column where we take the current market value for the shares and subtract the SP 500 value.
    merged_portfolio_sp_latest['Abs Value Compare'] = merged_portfolio_sp_latest['Ticker Share Value'] - merged_portfolio_sp_latest['SP 500 Value']
    
    # calculates profit / loss for stock position.
    merged_portfolio_sp_latest['Stock Gain / (Loss)'] = merged_portfolio_sp_latest['Ticker Share Value'] - merged_portfolio_sp_latest['Cost Basis']
    
    # calculates profit / loss for SP 500.
    merged_portfolio_sp_latest['SP 500 Gain / (Loss)'] = merged_portfolio_sp_latest['SP 500 Value'] - merged_portfolio_sp_latest['Cost Basis']
    
    # merge the DataFrame of the starting price (YTD = year to date)
    global merged_portfolio_sp_latest_YTD
    merged_portfolio_sp_latest_YTD = pd.merge(merged_portfolio_sp_latest, adj_close_start, on = 'Ticker')
    
    # delete the duplicate column
    del merged_portfolio_sp_latest_YTD['Date']
    
    # rename (and this Ticker start year close is based on the stocks_time_base())
    merged_portfolio_sp_latest_YTD.rename(columns={'Adj Close': 'Ticker Start Year Close'}, inplace = True)
    
    # add a column to change the date of 'Start of Year' into datastamp format
    for i in range(len(merged_portfolio_sp_latest_YTD)):
        merged_portfolio_sp_latest_YTD.at[i, 'datestamp_start_of_the_year'] = Changestamp(merged_portfolio_sp_latest_YTD['Start of Year'][i])
    
    # 'Start of Year' is a column of the excel sheet
    global merged_portfolio_sp_latest_YTD_sp
    merged_portfolio_sp_latest_YTD_sp = pd.merge(merged_portfolio_sp_latest_YTD, sp_500_adj_close_start, left_on = 'datestamp_start_of_the_year', right_on = 'datestamp2')
    
    del merged_portfolio_sp_latest_YTD_sp['Date']
    
    # renaming so that it's clear this column is SP 500 start of year close
    merged_portfolio_sp_latest_YTD_sp.rename(columns={'Adj Close': 'SP Start Year Close'}, inplace=True)
    
    # YTD return for portfolio position
    merged_portfolio_sp_latest_YTD_sp['Share YTD'] = merged_portfolio_sp_latest_YTD_sp['Ticker Adj Close'] / merged_portfolio_sp_latest_YTD_sp['Ticker Start Year Close'] - 1
    
    # YTD return for SP to run compares
    merged_portfolio_sp_latest_YTD_sp['SP 500 YTD'] = merged_portfolio_sp_latest_YTD_sp['SP 500 Latest Close'] / merged_portfolio_sp_latest_YTD_sp['SP Start Year Close'] - 1
    
    # alphabetical order
    merged_portfolio_sp_latest_YTD_sp = merged_portfolio_sp_latest_YTD_sp.sort_values(by = 'Ticker', ascending = True)
    
    # cumulative sum of original investment
    merged_portfolio_sp_latest_YTD_sp['Cum Invst'] = merged_portfolio_sp_latest_YTD_sp['Cost Basis'].cumsum()
    
    # cumulative sum of Ticker Share Value (latest FMV based on initial quantity purchased).
    merged_portfolio_sp_latest_YTD_sp['Cum Ticker Returns'] = merged_portfolio_sp_latest_YTD_sp['Ticker Share Value'].cumsum()
    
    # Cumulative sum of SP Share Value (latest FMV driven off of initial SP equiv purchase).
    merged_portfolio_sp_latest_YTD_sp['Cum SP Returns'] = merged_portfolio_sp_latest_YTD_sp['SP 500 Value'].cumsum()
    
    # Cumulative CoC multiple return for stock investments
    merged_portfolio_sp_latest_YTD_sp['Cum Ticker ROI Mult'] = merged_portfolio_sp_latest_YTD_sp['Cum Ticker Returns'] / merged_portfolio_sp_latest_YTD_sp['Cum Invst']
    
    portfolio_df.reset_index(inplace = True)
    
    global adj_close_acq_date
    adj_close_acq_date = pd.merge(adj_close, portfolio_df, on = 'Ticker')
    
    # delete useless columns
    del adj_close_acq_date['Quantity']
    del adj_close_acq_date['Unit Cost']
    del adj_close_acq_date['Cost Basis']
    del adj_close_acq_date['Start of Year']
    
    # sort by these columns in this order in order to make it clearer where compare for each position should begin.
    adj_close_acq_date.sort_values(by = ['Ticker', 'Acquisition Date', 'Date'], ascending = [True, True, True], inplace = True)
    
    # make 'Date' and 'Acquisition Date' into datestamp format, so that we can make subtraction to calculate which date is before which date
    for i in range(len(adj_close_acq_date)):
        adj_close_acq_date.at[i, 'Date_for_subtraction'] = Changestamp(adj_close_acq_date['Date'][i])
    
    for i in range(len(adj_close_acq_date)):
        adj_close_acq_date.at[i, 'Date_for_subtraction_acquisition'] = Changestamp(adj_close_acq_date['Acquisition Date'][i])
        
    # anything less than 0 means that the stock close was prior to acquisition.
    adj_close_acq_date['Date Delta'] = adj_close_acq_date['Date_for_subtraction'] - adj_close_acq_date['Date_for_subtraction_acquisition']
    
    adj_close_acq_date['Date Delta'] = adj_close_acq_date[['Date Delta']].apply(pd.to_numeric)  
    
    # Modified the dataframe being evaluated to look at highest close which occurred after Acquisition Date (aka, not prior to purchase)
    global adj_close_acq_date_modified
    adj_close_acq_date_modified = adj_close_acq_date[adj_close_acq_date['Date Delta'] >= 0]
    
    # find the max adjusted close, hope it has some meaning
    # this pivot table will index on the Ticker and Acquisition Date, and find the max adjusted close
    global adj_close_pivot
    adj_close_pivot = adj_close_acq_date_modified.pivot_table(index = ['Ticker', 'Acquisition Date'], values = 'Adj Close', aggfunc = np.max)
    adj_close_pivot.reset_index(inplace = True)
    
    # merge the adj close pivot table with the adj_close table in order to grab the date of the Adj Close High (good to know)
    global adj_close_pivot_merged
    adj_close_pivot_merged = pd.merge(adj_close_pivot, adj_close, on = ['Ticker', 'Adj Close'])
    
    # merge the Adj Close pivot table with the master dataframe to have the closing high since you have owned the stock
    global merged_portfolio_sp_latest_YTD_sp_closing_high
    merged_portfolio_sp_latest_YTD_sp_closing_high = pd.merge(merged_portfolio_sp_latest_YTD_sp, adj_close_pivot_merged
                                                 , on=['Ticker', 'Acquisition Date'])
    # renaming so that it's clear that the new columns are two year closing high and two year closing high date
    merged_portfolio_sp_latest_YTD_sp_closing_high.rename(columns = {'Adj Close': 'Closing High Adj Close', 'Date': 'Closing High Adj Close Date'}, inplace = True)
    
    #calculate the percentage difference between the highest price and the current price in the period of time starting from the start 
    # date and now, which is a good indicator to show if the investor is good at selling or not.
    merged_portfolio_sp_latest_YTD_sp_closing_high['Pct off High'] = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker Adj Close'] / merged_portfolio_sp_latest_YTD_sp_closing_high['Closing High Adj Close'] - 1 
    
    return merged_portfolio_sp_latest_YTD_sp_closing_high
#-----------------------------------------------------------------------------
# This part deals with Graphing
# When investor wants to se some certain kind of graph, he just needs to input graph1() or graph2() and he will get the graph in interactive html format.

# The first graph is to compare YTD (year to date, the return rate of buying the 
# stock at the beginning of this year and selling it now) returns of investor's stocks and S&P 500
def graph1():
    trace1 = go.Bar(
        x = merged_portfolio_sp_latest_YTD_sp['Ticker'][0:10],
        y = merged_portfolio_sp_latest_YTD_sp['Share YTD'][0:10],
        name = 'Ticker YTD')
    
    trace2 = go.Scatter(
        x = merged_portfolio_sp_latest_YTD_sp['Ticker'][0:10],
        y = merged_portfolio_sp_latest_YTD_sp['SP 500 YTD'][0:10],
        name = 'SP500 YTD')
        
    data = [trace1, trace2]
    
    layout = go.Layout(title = 'YTD Return vs S&P 500 YTD'
        , barmode = 'group'
        , yaxis=dict(title='Returns', tickformat=".2%")
        , xaxis=dict(title='Ticker')
        , legend=dict(x=.8,y=1)
        )
    
    global fig1
    fig1 = go.Figure(data = data, layout = layout)
    plot(fig1)


# Current share price versus closing high (the highest price) since purchased,
# which can show if this investor is good at selling his stocks at the best time
def graph2():
    trace1 = go.Bar(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['Pct off High'][0:10],
        name = 'Pct off High')
        
    data = [trace1]
    
    layout = go.Layout(title = 'Adj Close % off of High'
        , barmode = 'group'
        , yaxis=dict(title='% Below Adj Close High', tickformat=".2%")
        , xaxis=dict(title='Ticker')
        , legend=dict(x=.8,y=1)
        )
    fig2 = go.Figure(data=data, layout=layout)
    plot(fig2)


# Total return comparison charts
def graph3():
    trace1 = go.Bar(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['ticker return'][0:10],
        name = 'Ticker Total Return')
    
    trace2 = go.Scatter(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['SP Return'][0:10],
        name = 'SP500 Total Return')
        
    data = [trace1, trace2]
    
    layout = go.Layout(title = 'Total Return vs S&P 500'
        , barmode = 'group'
        , yaxis=dict(title='Returns', tickformat=".2%")
        , xaxis=dict(title='Ticker', tickformat=".2%")
        , legend=dict(x=.8,y=1)
        )
    fig3 = go.Figure(data=data, layout=layout)
    plot(fig3)


# Cumulative returns over time
def graph4():
    trace1 = go.Bar(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['Stock Gain / (Loss)'][0:10],
        name = 'Ticker Total Return ($)')
    
    trace2 = go.Bar(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['SP 500 Gain / (Loss)'][0:10],
        name = 'SP 500 Total Return ($)')
    
    trace3 = go.Scatter(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['ticker return'][0:10],
        name = 'Ticker Total Return %',
        yaxis='y2')
    
    data = [trace1, trace2, trace3]
    
    layout = go.Layout(title = 'Gain / (Loss) Total Return vs S&P 500'
        , barmode = 'group'
        , yaxis=dict(title='Gain / (Loss) ($)')
        , yaxis2=dict(title='Ticker Return', overlaying='y', side='right', tickformat=".2%")
        , xaxis=dict(title='Ticker')
        , legend=dict(x=.75,y=1)
        )
    fig4 = go.Figure(data=data, layout=layout)
    plot(fig4)


# Total cumulative investment over time
def graph5():
    trace1 = go.Bar(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Invst'],
        # mode = 'lines+markers',
        name = 'Cum Invst')
    
    trace2 = go.Bar(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum SP Returns'],
        # mode = 'lines+markers',
        name = 'Cum SP500 Returns')
    
    trace3 = go.Bar(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Ticker Returns'],
        # mode = 'lines+markers',
        name = 'Cum Ticker Returns')
    
    trace4 = go.Scatter(
        x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
        y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Ticker ROI Mult'],
        # mode = 'lines+markers',
        name = 'Cum ROI Mult'
        , yaxis='y2')
    
    data = [trace1, trace2, trace3, trace4]
    
    layout = go.Layout(title = 'Total Cumulative Investments Over Time'
        , barmode = 'group'
        , yaxis=dict(title='Returns')
        , xaxis=dict(title='Ticker')
        , legend=dict(x=.4,y=1)
        , yaxis2=dict(title='Cum ROI Mult', overlaying='y', side='right')               
        )
    fig5 = go.Figure(data=data, layout=layout)
    plot(fig5)