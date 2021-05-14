# -*- coding: utf-8 -*-
# The stock interface stage
import tkinter as tk
import datetime

from class_input_investor_tickers import judge_whether_ticker_is_legal
from class_input_investor_tickers import judge_whether_ticker_acquisition_is_legal_in_format
from class_input_investor_tickers import judge_whether_ticker_acquisition_is_legal_in_weekdays
from class_input_investor_tickers import judge_whether_ticker_quantity_is_legal
from class_input_investor_tickers import judge_whether_ticker_unit_cost_is_legal
from class_input_investor_tickers import dataframe_initialization
from class_input_investor_tickers import change_acquisition_date_format

from input_process import investor_please_input
from input_process import graph1
from input_process import graph2
from input_process import graph3
from input_process import graph4
from input_process import graph5

window = tk.Tk()
window.title('investor please input something')
window.geometry('1200x850')

#-------------------------------------------------------------------
'''
The aim of this file is to generate an interface for investor to input his portfolio information
so that later we can use the input information to do performance analysis and show the visualizations.
'''
#-------------------------------------------------------------------
# initialization
# def interface_window()
def input_in_interface():
    global ii

    # input ticker name button
    l_ticker_name = tk.Label(window, text='please input your ticker name', bg='Silver', font=('Arial', 12), width=30, height = 2)
    l_ticker_name.pack()
    
    e_ticker_name = tk.Entry(window)
    e_ticker_name.pack()
    #-------------------------
    # input ticker acquisition date button
    
    l_acquisition_date = tk.Label(window, text='please input your acquisition date in form "yyyy-mm-dd"', bg='Silver', font=('Arial', 12), width=50, height = 2)
    l_acquisition_date.pack()
    
    e_acquisition_date = tk.Entry(window)
    e_acquisition_date.pack()
    #-------------------------
    # input quantity button
    l_quantity = tk.Label(window, text='please input the number of shares you bought!', bg='Silver', font=('Arial', 12), width=46, height = 2)
    l_quantity.pack()
    
    e_quantity = tk.Entry(window)
    e_quantity.pack()
    #-------------------------
    # input cost
    l_cost = tk.Label(window, text='please input your cost for each share!', bg='Silver', font=('Arial', 12), width=46, height = 2)
    l_cost.pack()
    
    e_cost = tk.Entry(window)
    e_cost.pack()
#-------------------------
    # define a function to check if all the information input by customers is in their legal form
    def input_total_correctly():
        global ii
        global var_ticker_name
        var_ticker_name = e_ticker_name.get()
        global var_acquisition_date
        var_acquisition_date = e_acquisition_date.get()    
        global var_quantity
        var_quantity = e_quantity.get()    
        global var_cost
        var_cost = e_cost.get()
        
        check1 = judge_whether_ticker_is_legal(var_ticker_name)
        check2 = judge_whether_ticker_acquisition_is_legal_in_format(var_acquisition_date)
        check3 = judge_whether_ticker_acquisition_is_legal_in_weekdays(var_acquisition_date)
        check4 = judge_whether_ticker_quantity_is_legal(var_quantity)
        check5 = judge_whether_ticker_unit_cost_is_legal(var_cost)
        
        global check_all
        check_all = check1 and check2 and check3 and check4 and check5
        
        if (check1 and check2 and check3 and check4 and check5):
            l_total.config(text='Your inputs are all legal! Please click "save" button to save your information!')
        elif check1==False:
            l_total.config(text='Your ticker name is not found! Correct your input!')
        elif check2==False:
            l_total.config(text='Your acquisition date format is illegal! Correct your input!')
        elif check3==False:
            l_total.config(text='Your acquisition date is on weekend! Correct your input!')
        elif check4==False:
            l_total.config(text='Your quantity is illegal! Correct your input!')
        elif check5==False:
            l_total.config(text='Your unit cost is illegal! Correct your input!')        
            
    b_total_button = tk.Button(window, text='Finish Input!', font=('Arial', 12), width=30, height=2, command=input_total_correctly)
    b_total_button.pack()
    
    #------------------
    # add a input reminder
    l_total = tk.Label(window, text='In this box you will see if your input is legal', bg='FloralWhite', font=('Arial', 12), width=70, height = 2)
    l_total.pack()
    
    # save button, after pressing it the input information would be saved for further use
    def save_info():
        global ii
        if var_ticker_name and var_acquisition_date and var_quantity and var_cost == '':
            l_total.config(text='You must input something! Correct your input!')
        else:
            if check_all == True:
                global ii
                global data_base
                global transfer_ticker_name
                global transfer_quantity
                global transfer_acquisition_date
                global transfer_cost
                
                transfer_ticker_name = var_ticker_name
                transfer_acquisition_date = var_acquisition_date
                transfer_quantity = var_quantity
                transfer_cost = var_cost
                
                transfer_acquisition_date = change_acquisition_date_format(transfer_acquisition_date)
                global data
                data = {}
                data['Acquisition Date'] = transfer_acquisition_date
                data['Ticker'] = transfer_ticker_name
                data['Quantity'] = transfer_quantity
                data['Unit Cost'] = transfer_cost
                transfer_cost_basis = float(transfer_quantity)* float(transfer_cost)
                data['Cost Basis'] = transfer_cost_basis
                data['Start of Year'] = datetime.datetime(2019,12,31)
                data_base.loc[ii] = data
                
                l_total.config(text='Your information has been saved!')
                def ask_yes_no():
                    global whether_input_more
                    global ii
                    return tk.messagebox.askyesno(title='Hi', message='Do you want to input more?')
                
                # see if the customer wants to input more, if not generate the graphs, 
                # if yes, continue inputting.
                whether_input_more = ask_yes_no()
                if whether_input_more == True:
                    ii += 1
                    # delete all the labels and buttons
                    l_ticker_name.forget()
                    e_ticker_name.forget()
                    l_acquisition_date.forget()
                    e_acquisition_date.forget()
                    l_quantity.forget()
                    e_quantity.forget()
                    l_cost.forget()
                    e_cost.forget()
                    b_total_button.forget()
                    l_total.forget()
                    b_save.forget()
                    
                    input_in_interface()
                else:
                    investor_please_input(data_base)
                    
                    # add the graph module here, we have 5 graphs so we added 5 buttons and 5 explanations above.
                    b_graph1 = tk.Button(window, text='Show graph1', font=('Arial', 10), width=20, height=1, command=graph1)
                    b_graph2 = tk.Button(window, text='Show graph2', font=('Arial', 10), width=20, height=1, command=graph2)
                    b_graph3 = tk.Button(window, text='Show graph3', font=('Arial', 10), width=20, height=1, command=graph3)
                    b_graph4 = tk.Button(window, text='Show graph4', font=('Arial', 10), width=20, height=1, command=graph4)
                    b_graph5 = tk.Button(window, text='Show graph5', font=('Arial', 10), width=20, height=1, command=graph5)
                    
                    l_graph1 = tk.Label(window, text="Graph1: Compare stock's YTD with S&P 500", bg='FloralWhite', font=('Arial', 10), width=70, height = 1)
                    l_graph2 = tk.Label(window, text="Graph2: Compare current price with highest price since purchased", bg='FloralWhite', font=('Arial', 10), width=70, height = 1)
                    l_graph3 = tk.Label(window, text="Graph3: Total return comparison charts", bg='FloralWhite', font=('Arial', 10), width=70, height = 1)
                    l_graph4 = tk.Label(window, text="Graph4: Cumulative returns over time", bg='FloralWhite', font=('Arial', 10), width=70, height = 1)
                    l_graph5 = tk.Label(window, text="Graph5: Total cumulative investment over time", bg='FloralWhite', font=('Arial', 10), width=70, height = 1)
                    
                    l_graph1.pack()
                    b_graph1.pack()
                    l_graph2.pack()
                    b_graph2.pack()
                    l_graph3.pack()
                    b_graph3.pack()
                    l_graph4.pack()
                    b_graph4.pack()
                    l_graph5.pack()
                    b_graph5.pack()
                    
                    tk.Button(window, text='quit', command=window.quit).pack()
                    return data_base
                
            else:
                l_total.config(text='Go click the finish input button first!')
    
    b_save = tk.Button(window, text='Save and proceed!', font=('Arial', 12), width=40, height=2, command=save_info)
    b_save.pack()
    
    window.mainloop()

# initialize data_base
ii = 1
# database is for further use by other files.
data_base = dataframe_initialization()
def investor_interface_input():
    global data_base
    global ii
    input_in_interface()
    return data_base