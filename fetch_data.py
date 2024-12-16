import os
import pickle
import time
import pandas as pd
import yfinance as yf
from fredapi import Fred
#from alpha_vantage.fundamentaldata import FundamentalData
from dotenv import load_dotenv
import pandas_ta as ta

load_dotenv()

# Set API keys
#ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FRED_API_KEY= "4c871095fcbd6fe0ab70188de75236ab"

#if not ALPHA_VANTAGE_API_KEY:
#    raise ValueError("Set Alpha Vantage API key in the '.env' file.")
if not FRED_API_KEY:
    raise ValueError("Set FRED API key in the '.env' file.")

fred = Fred(api_key=FRED_API_KEY)


START_DATE = '2017-01-01'
END_DATE = '2024-11-02'


COMPANIES = {
    'Telus': ['T.TO'],
    'Rogers Communications': ['RCI-B.TO'],
    'Quebecor': ['QBR-B.TO'],
    'Cogeco Communications': ['CCA.TO'],
    'Bell Canada (BCE Inc.)': ['BCE.TO'],
    'Hydro One': ['H.TO'],
    'Fortis': ['FTS.TO'],
    'AltaGas': ['ALA.TO'],
    'Canadian National Railway': ['CNR.TO'],
    'Canadian Pacific Railway': ['CP.TO'],
    'Metro': ['MRU.TO'],
    'Loblaws': ['L.TO'],
    'Empire': ['EMP-A.TO'],
    'Alimentation Couche-Tard': ['ATD.TO'],
}


MACRO_INDICATORS = {
    'Real GDP': {'series_id': 'NGDPRSAXDCCAQ', 'frequency': 'Quarterly'},
    'GDP Growth Annual': {'series_id': 'CANGDPRQPSMEI', 'frequency': 'Annual'},
    'GDP Growth Quarterly': {'series_id': 'CANGDPRAPSMEI', 'frequency': 'Quarterly'},
    'CPI Growth Monthly': {'series_id': 'CPALTT01CAM659N', 'frequency': 'Monthly'},
    'CPI Growth Quarterly': {'series_id': 'CPALTT01CAQ657N', 'frequency': 'Quarterly'},
    'PPI Growth Monthly': {'series_id': 'CANPIEAMP01GPM', 'frequency': 'Monthly'},
    'PPI Growth Quarterly': {'series_id': 'CANPIEAMP01GPQ', 'frequency': 'Quarterly'},
    'Unemployment Rate': {'series_id': 'LRUNTTTTCAQ156S', 'frequency': 'Quarterly'},
    'Interest Rate': {'series_id': 'IRSTCB01CAM156N', 'frequency': 'Monthly'},
    'Industrial Production Ex Construction': {'series_id': 'PRINTO01CAQ189S', 'frequency': 'Quarterly'},
    'Industrial Production Growth Ex Construction': {'series_id': 'PRINTO01CAA657S', 'frequency': 'Annual'},
    'Industrial Production: Construction': {'series_id': 'PRINTO01CAQ189S', 'frequency': 'Quarterly'},
    'Industrial Production Growth: Construction': {'series_id': 'PRINTO01CAA657S', 'frequency': 'Annual'},
    'Consumer Confidence': {'series_id': 'CSCICP03CAM665S', 'frequency': 'Monthly'},
    'Retail Sales Growth': {'series_id': 'SLRTTO01CAA657S', 'frequency': 'Quarterly'},
    'Housing Starts': {'series_id': 'HSN1F', 'frequency': 'Monthly'},
    'Business Confidence': {'series_id': 'CANBSCICP02STSAQ', 'frequency': 'Quarterly'},
}


def get_technical_data(companies, start_date, end_date):
    technical_data_list = []
    for company, tickers in companies.items():
        data_fetched = False
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    continue
                data.reset_index(inplace=True)
                data.columns = data.columns.get_level_values(0)
                # Calculate returns
                data['Daily Return'] = data['Close'].pct_change()
                data['Adjusted Daily Return'] = data['Adj Close'].pct_change()
                # Add company and ticker info
                data['Company'] = company
                data['Ticker'] = ticker
                # Calculate technical indicators
                data = calculate_technical_indicators(data)
                technical_data_list.append(data)
                print(f"Technical data fetched for {company} ({ticker})")
                data_fetched = True
                break
            except Exception as e:
                print(f"Error fetching technical data for {company} ({ticker}): {e}")
        if not data_fetched:
            print(f"Failed to fetch technical data for {company}")
    # Combine all data into one DataFrame
    technical_data = pd.concat(technical_data_list, ignore_index=True)
    return technical_data

def calculate_technical_indicators(data):
    # Moving Averages
    data['SMA_20'] = ta.sma(data['Adj Close'], length=20)
    data['SMA_100'] = ta.sma(data['Adj Close'], length=100)
    data['EMA_20'] = ta.ema(data['Adj Close'], length=20)
    data['EMA_100'] = ta.ema(data['Adj Close'], length=100)
    # Bollinger Bands
    bbands = ta.bbands(data['Adj Close'], length=20)
    data = pd.concat([data, bbands], axis=1)
    # Relative Strength Index
    data['RSI_14'] = ta.rsi(data['Adj Close'], length=14)
    # Moving Average Convergence Divergence
    macd = ta.macd(data['Adj Close'])
    data = pd.concat([data, macd], axis=1)
    return data

def melt_technical_data(data):
    """
    Melt the technical data DataFrame into long format.
    """
    id_vars = ['Company', 'Ticker', 'Date']
    value_vars = [col for col in data.columns if col not in id_vars]
    melted_data = data.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Variable',
        value_name='Value'
    )
    return melted_data


technical_data = get_technical_data(COMPANIES, START_DATE, END_DATE)
technical_data['Date'] = pd.to_datetime(technical_data['Date']).dt.date
technical_data = technical_data[technical_data['Date'] >= pd.to_datetime('2018-01-01').date()]
technical_melted = melt_technical_data(technical_data)


def get_macro_data(indicators, start_date, end_date):
    macro_data_list = []
    for indicator_name, info in indicators.items():
        series_id = info['series_id']
        frequency = info['frequency']
        try:
            data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            data = data.to_frame(name='Value')
            data['Indicator'] = indicator_name
            data['Frequency'] = frequency
            data.index.rename('Date', inplace=True)
            data.reset_index(inplace=True)
            macro_data_list.append(data)
            
            print(f"Macro data fetched for {indicator_name}")
        except Exception as e:
            print(f"Error fetching macro data for {indicator_name} ({series_id}): {e}")

    # Combine all data into one DataFrame
    macro_data = pd.concat(macro_data_list, ignore_index=True)
    return macro_data

macro_data = get_macro_data(MACRO_INDICATORS, START_DATE, END_DATE)


def get_recent_info(companies):
   tickers_data= {}
   for company, tickers in companies.items():
      for ticker in tickers:
         ticker_object = yf.Ticker(ticker)
         #convert info() output from dictionary to dataframe
         temp = pd.DataFrame.from_dict(ticker_object.info, orient="index")
         temp.reset_index(inplace=True)
         temp['Company'] = company
         temp.columns = ["Attribute", "Recent", "Company"]
         
         # add (ticker, dataframe) to main dictionary
         tickers_data[ticker] = temp

   combined_data = pd.concat(tickers_data)
   combined_data = combined_data.reset_index()
   del combined_data["level_1"] 
   combined_data.columns = ["Ticker", "Attribute", "Recent", "Company"] 
   return combined_data


actual_data = get_recent_info(COMPANIES)


# Save the outputs as pickle files
with open(os.path.join('data', 'tabular_data', 'macro_data.pkl'), 'wb') as macro_file:
   pickle.dump(macro_data, macro_file)

with open(os.path.join('data', 'tabular_data', 'technical_data.pkl'), 'wb') as technical_file:
   pickle.dump(technical_data, technical_file)

with open(os.path.join('data', 'tabular_data', 'technical_melted.pkl'), 'wb') as melted_file:
   pickle.dump(technical_melted, melted_file)

with open(os.path.join('data', 'tabular_data', 'actual_data.pkl'), 'wb') as actual_file:
    pickle.dump(actual_data, actual_file)