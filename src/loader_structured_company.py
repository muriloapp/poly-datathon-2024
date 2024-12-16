###
###     Script for loading the structured data into db
###

from functions import *

###     database connection

import psycopg

conn = psycopg.connect("postgresql://janos:(q0)r:0|QL>N[}b>>PA<6Eu6fE5x@database-3.cluster-czkmkismw47i.us-west-2.rds.amazonaws.com/datathon_db")

c = conn.cursor()

###     getting data and writing to db

import pandas as pd
import pandas.io.sql as sqlio
import yfinance as yf
import pandas_ta as ta
from io import StringIO


metadata = get_table("metadata", conn)

# actual loading
for t in metadata["yf_ticker"]:
    load_technical(t, conn)


###     exit
conn.commit()
conn.close()
