# -*- coding: utf-8 -*-
"""
Created on Tue Apr 7 11:37:52 2020

@author: mario and zi chao
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import math
import sys

ticker = input("Which S&P500 stock is to be studied? Input a capitalized ticker. Recommended: AMZN, NVDA ")
#Note: the model is exponential and the steps are defined so stocks with really big prices (eg. GOOG) or stocks 
#that don't really fit an exponential model will obviously not be great choices
#The recommeneded stocks are the ones used to perfect the model and the ones that seem to work best, 
#others might take a while but will nevertheless be found



### Read any of the S&P 500 csv datafiles, path to be edited to find the kaggle data folder, mine was on desktop
try:
    stock = pd.read_csv("C:/Users/mario/Desktop/sandp500/individual_stocks_5yr/individual_stocks_5yr/"+ticker+"_data.csv")
except:
    #Exception exit
    sys.exit("Stock "+ ticker + " not found.")
# Adding an index column and singling out columns

index = []
for i in range(0, len(stock.index)):
    index.append(i)
#Splitting the data into pre 2017 and 2017 to allow us to try to predict 2017
stock["Index"] = index
indexstore = index
stock["date"] = pd.to_datetime(stock["date"])
stock["year"] = stock["date"].dt.year
stockstore = stock

stock2017 = stock.query("year == 2017")
stock = stock.query("year != 2018").query("year != 2017")

print(stock.tail())
print(stock2017.head())
close = stock["close"]
index = stock["Index"]
index2017 = stock2017["Index"]
close2017 = stock2017["close"]
# Initial stock chart 

plt.plot(index, close)
plt.xlabel("Cumulative Days")
plt.ylabel("Stock Price ($)")
plt.show()

# END OF DATA 


#Defining the basic exponential model
def expom(x,a,b,c):
    return a*((math.e)**(b*x)) + c


#Defining a function to quantify the error of the model (average distance between prediction and data)
def costfunction(x,y,g):
    yp = np.empty(len(x))
    error = np.empty(len(x))
    for i in range(len(x)):
        yp[i] = expom(x[i], g[0], g[1], g[2])
        error[i] = (yp[i] - y[i])**2
        
    cost = (1.0/len(x))*np.sum(error)
    return cost

#Defining a function to determine if the model overshoots or undershoots the data
def greaterlesser(x, y, g):
    yp = np.empty(len(x))
    error = np.empty(len(x))
    for i in range(len(x)):
        yp[i] = expom(x[i], g[0], g[1], g[2])
        error[i] = (yp[i] - y[i])
    
    return np.sum(error)

#Defining a function that converts cost into normalized root mean square error
def normalizedRMSE(cost):
    NRMSE = math.sqrt(cost)/(close.max() - close.min())
    return NRMSE

# Used to automate the guess of parameter 'c'
initial_close = stock.loc[stock['Index'] == 0]['close']

try:
    iterations = int(input("How many iterations are to be run? ")) #Number of iterations of the fitting algorithm with randomized starting guesses
    #Recommended 10 for a better result, 1 for testing. Obviously, more = precision
except:
    #Exception exit
    sys.exit("Invalid number of iterations")
    #NOTE: High number of interations WILL take time and some guesses are worse than others and will take more time

guessco = [0,0,0] #Storage for the best guess parameters
costco = 10**10    #Storage for the best cost found with the above parameters, starts with a large value cuz 
#the algo compares last and stored and stores the lesser

for i in range(iterations):
    print(str(i + 1)+ "/" + str(iterations) + " Iterations") 
    #Randomized guesses within a range for the parameters, ranges modeled for the recommended stocks mentionned
    guess = [0.01*np.random.randint(50,200),
             0.0001*np.random.randint(1,5),
             np.random.randint(0.5*initial_close, 2.5*initial_close)]
    
    costnow = costfunction(index, close, guess)
    
    
    storage = [10**11, 10**10] 
    #Storage of costs starts with 2 decreasing large values as this is the condition for continued
    #modification of the guesses
    
    
    alpha = [0.1, 0.00001, 1] #Size of steps for the iterative process of the guesses
    print("b")
    while storage[-1] < storage[-2]:
        
        print(costnow)
                  
        
        if greaterlesser(index, close, guess) > 0: #If data is overshot, decrease the parameter
            guess[1] = guess[1] - alpha[1]
        else:                                       #Else increase the parameter
            guess[1] = guess[1] + alpha[1]
                
        costnow = costfunction(index, close, guess)
        storage.append(costnow)   
        
        
    storage = [10**11, 10**10]
    
    print("a")
    while storage[-1] < storage[-2]:
        counter = 0
        print(costnow)
                
        if greaterlesser(index, close, guess) > 0:
            guess[0] = guess[0] - alpha[0]
        else:
            guess[0] = guess[0] + alpha[0]
                
        costnow = costfunction(index, close, guess)
        storage.append(costnow)   
        counter += 1
        if counter >= 500:
            break
        
    storage = [10**11, 10**10]
    print("c")
    while storage[-1] < storage[-2]:
        
        print(costnow)
                
        if greaterlesser(index, close, guess) > 0:
            guess[2] = guess[2] - alpha[2]
        else:
            guess[2] = guess[2] + alpha[2]
                
        costnow = costfunction(index, close, guess)
        storage.append(costnow)  
        
        
    if costnow <= costco: #If the cost found is better than the one stored, update the stored values
        
        costco = costnow
        guessco = guess

#array with y as a function of x using the expom and our guesses   
solutionplot = np.empty(len(indexstore)) 

for i in range(len(indexstore)):
    solutionplot[i] = expom(indexstore[i], guessco[0], guessco[1], guessco[2])


#Final Cost including all the data (2017 and all)
fcost = costfunction(indexstore, stockstore["close"], guessco)
#Final plot
print('Final guess list', guessco)
print('Final cost', fcost)
print("Final cost with fitting data", costco)
print('NRMSE', normalizedRMSE(fcost))
plt.title("Best Fitting Exponential Curve of " + ticker)
plt.plot(close, label = "Pre-2017 Close")
plt.plot(solutionplot, label = "Prediction")
plt.plot(close2017, label = "2017 Close")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel("Cumulative Days")
plt.ylabel("Stock Price ($)")
plt.show()    


