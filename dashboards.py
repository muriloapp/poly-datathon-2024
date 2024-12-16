import plotly.graph_objects as go
import pandas as pd
import pickle
import streamlit as st

import plotly.graph_objects as go

def plot_stock_with_indicators(data, ticker):
    """
    Plot stock data with multiple indicators and volume on a secondary y-axis.

    Parameters:
    - data: DataFrame containing 'Date', 'Ticker', 'Variable', and 'Value' columns.
    - ticker: String representing the stock ticker to filter the data.
    """
    # Filter data for the selected ticker
    ticker_df = data[data['Ticker'] == ticker]

    # List of indicators to include in the chart, each with a distinct color
    indicators = {
        "SMA_20": "green",
        #"EMA_20": "green",
        # # "EMA_100": "orange",
        # # "SMA_100": "purple"
        "SMA_100": "red",
        # "BBU_20_2.0" : "green",
        # "BBL_20_2.0" : "green",
    }

    # Initialize the figure
    fig = go.Figure()

    # Plot the Adjusted Close prices as the main line
    fig.add_trace(go.Scatter(
        x=ticker_df[ticker_df["Variable"] == "Adj Close"]["Date"],
        y=ticker_df[ticker_df["Variable"] == "Adj Close"]["Value"],
        mode='lines',
        name='Adj Close',
        line=dict(width=2, color="black")
    ))

    # Add traces for each indicator with different colors
    for indicator, color in indicators.items():
        fig.add_trace(go.Scatter(
            x=ticker_df[ticker_df["Variable"] == indicator]["Date"],
            y=ticker_df[ticker_df["Variable"] == indicator]["Value"],
            mode='lines',
            name=indicator,
            line=dict(width=1.5, color=color),
            visible=True  # Make all indicators visible by default
        ))

    # Add a bar plot for Volume on a secondary y-axis
    fig.add_trace(go.Bar(
        x=ticker_df[ticker_df["Variable"] == "Volume"]["Date"],
        y=ticker_df[ticker_df["Variable"] == "Volume"]["Value"],
        name='Volume',
        opacity=0.5,
        marker=dict(color="darkgray"),
        yaxis='y2'
    ))

    # Update layout with title, axis labels, secondary y-axis, and margin adjustments
    fig.update_layout(
        title=f'Stock Analysis for {ticker_df["Company"].iloc[0]} ({ticker})',  
        xaxis_title='Date',
        yaxis=dict(title='Price', side='left'),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, ticker_df[ticker_df['Variable'] == 'Volume']['Value'].max() * 1.2]
        ),
        legend_title='Variables',
        width=1200,       # Set plot width
        height=600,       # Set plot height
        margin=dict(l=20, r=20, t=50, b=20),  # Reduce whitespace at the margins
    )

    # Display the interactive chart
    st.plotly_chart(fig, use_container_width=True)

def display_stock_info(data, filtered_data, selected_ticker):
    sector = data.loc[data['Ticker'] == selected_ticker, 'Sector'].values[0]
    sector_data = filtered_data[filtered_data['Sector'] == sector]
    #sector_data['sector'] = sector
    sector_table = sector_data.pivot(index=['Company', 'Ticker'], columns='Attribute', values='Recent')
    sector_table["sector"] = sector
    #print(f"\nStock and Peers Information for {selected_ticker}:")
    return sector_table

# Example usage
#plot_stock_with_indicators(data, ticker='T.TO')