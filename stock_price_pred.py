import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
from datetime import date, timedelta


## Adding Configuration details for specifying date range of the stock price

NAME_OF_STOCK   = 'TSLA'
TILL_DATE       = date.today() - timedelta(days=1)
START_DATE      = date.today() - timedelta(days=90)


## Download Data Sets

tsla_df = yf.download(NAME_OF_STOCK,
                      start=START_DATE,
                      end=TILL_DATE,
                      progress=True).reset_index()
print(tsla_df.head())
print(tsla_df.tail())