This is a real-time, everyday-updated stock performance analyzer, stocke recommendation adviser with beautiful interactive interfaces.
（BTW it's an in-class big homework based project, if there are anything wrong with the codes or the package I am using, please contact me)

Before all, in order to make the files running smoothly, you need to download the packages in the 'requirements.txt' file by using the command: 
"pip install -r requirements.txt" in the file location in command.

The following is the function manual:

How to use this:
There are in total 6 files. 
I will give each of them a number:
	No1: stage1&2_investment_performance_analysis_and_visualization.py
	No2: stage3_stock_interface_recommendation.py
	No3: recommendation.py
	No4: input_process.py
	No5: stock_interface.py
	No6: class_input_investor_tickers.py

Firstly, if you want to input your ivestment information and see the beautiful stock performance, you need to run the No.1 file and input as ordered.
	After inputing your basic information and clicked "processed", you will see there are 5 options at the bottom of the window, and you can click on 
	whichever you want to see. After clicking the button, the visualization will appear in your browser.
	
Secondly, if you want to ask AI if your stock is worth buying or not, you can run the No.2 file and after waiting for around 1 min you will get an 
	advice from AI telling you your input stocks are worth buying or not based on basic machine learning techniques.

Two Warnings:
1. Due to the instability of the yfinance and pandas_datareader (through which all the raw stock information was retrieved), sometimes (around 1 in 10 
	times there will be some "HTTPError Service" unavailable.
	If that happens, you just need to run it again, and usually it will work.
2. When you are running the No.2 file, it might take as long as one minute to download all the necessary data in order to generate and 
	train the model and give you AI's opinion. Therefore when there is no error message and no response after running the file, please be patient,
	and if you still cannot see anything happening, just press "Blank" and you will see a window out. 
	
	Estimated running time:
		10 seconds to run No.1 file
		1 minute to run No.2 file
		1 second to run others

Enjoy your investment journey.
