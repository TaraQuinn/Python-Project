import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import math

import pandas as pd




"""""""""""Closing Price Graph"""""""""""
#For the purpose of simplicity, we are going to just consider one column 'Adj Close'
# And as a part of cleaning the dataset, we are going to drop rows with null values in them.
df = pd.read_csv(r'AAPL.csv',index_col='Date',parse_dates=True)
df=df.dropna()  #removes null values
df=df['Adj Close']['2021-11-10':'2022-11-09'] #to plot close prices bewtween these dates
df=pd.DataFrame(df)
plt.figure(figsize=(14,7))
plt.plot(df)
plt.show()





"""""""""""""Moving Average"""""""""""""

# Add a simple moving average
df['SMA_10'] = df['Adj Close'].rolling(window=10).mean()
# print the first 15 rows of data
print(df.head(15))
df['SMA_10'].plot(grid=True, figsize=(8,5))
plt.show()

AAPL = pd.read_csv(r'AAPL.csv',index_col='Date',parse_dates=True)
AAPL['Adj Close'] = AAPL['Adj Close']
AAPL['42d'] = np.round(AAPL['Adj Close'].rolling(window=42).mean(),2)
AAPL['20d'] = np.round(AAPL['Adj Close']. rolling(window=20).mean(),2)
AAPL['10d'] = np.round(AAPL['Adj Close']. rolling(window=10).mean(),2)

#plot moving average
AAPL[['Adj Close', '42d', '20d', '10d']].plot(grid=True, figsize=(8,5))
plt.show()
