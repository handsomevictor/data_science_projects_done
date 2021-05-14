# -*- coding: utf-8 -*-
# The stock recommendation interface stage
import tkinter as tk
from recommendation import machine_learning_to_recommend_buy_or_not
from class_input_investor_tickers import judge_whether_ticker_is_legal

window = tk.Tk()
window.title('investor please input the stock you want me to determine buy or not')
window.geometry('800x400')

def recommendation_interface():
    
    l_ticker_recommendation_input = tk.Label(window, text='please input the ticker name you want for advice', bg='Silver', font=('Arial', 12), width=56, height = 2)
    l_ticker_recommendation_input.pack()
    
    e_ticker_name_for_recommendation = tk.Entry(window)
    e_ticker_name_for_recommendation.pack()
    
    global l_total_recommendation
    l_total_recommendation = tk.Label(window, text='Here you will see all the responses', bg='Silver', font=('Arial', 12), width=56, height = 2)
    l_total_recommendation.pack()
    
    def save_recommendation_input():
        global var_ticker_name_recommendation
        var_ticker_name_recommendation = e_ticker_name_for_recommendation.get()
    
    b_button_save_the_input_information = tk.Button(window, text='First click here to save the input!', font=('Arial', 12), width=40, height=2, command=save_recommendation_input)
    b_button_save_the_input_information.pack() 
   
    def check_recommendation_name_function():
        global for_convenience
        check_recommendation_name = judge_whether_ticker_is_legal(var_ticker_name_recommendation)
        if check_recommendation_name:
            for_convenience = 1
            l_total_recommendation.config(text='Your inputs are legal! Please click the last button to show the result!')
        else:
            for_convenience = 0
            l_total_recommendation.config(text='Your ticker is not found! Please input something else!')
        return

    b_button_recommendation_check_name = tk.Button(window, text='Second click here to see if the input is correct!', font=('Arial', 12), width=40, height=2, command=check_recommendation_name_function)
    b_button_recommendation_check_name.pack()
    
    b_button_recommendation = tk.Button(window, text='Third click here to see the result!', font=('Arial', 12), width=40, height=2, command=show_recommendation_result)
    b_button_recommendation.pack()
    
    window.mainloop()

# seperate this function, because this function will take some time to run
def show_recommendation_result():
    global var_ticker_name_recommendation
    if for_convenience == 1:
        l_total_recommendation.config(text=machine_learning_to_recommend_buy_or_not(var_ticker_name_recommendation))
    else:
        l_total_recommendation.config(text='Error! Please check the former steps!')

recommendation_interface()

