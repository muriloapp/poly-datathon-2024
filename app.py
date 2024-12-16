import streamlit as st
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
import fitz  # PyMuPDF
import warnings
from dashboards import plot_stock_with_indicators, display_stock_info
from utils import ai_financial_assistant, query_llm, format_chat_history, ai_financial_assistant_chatbot
import boto3

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Financial AI Assistant",
    layout = "wide",
)

##### READ THE DATA #####
company_tickers = {
    "Couche-Tard": "ATD.TO",
    "Empire": "EMP-A.TO",
    "Loblaws": "L.TO",
    "Métro": "MRU.TO",
    "Canadian National Railway Company": "CNR.TO",
    "Canadian Pacific Kansas City": "CP.TO",
    "AltaGas": "ALA.TO",
    "Fortis": "FTS.TO",
    "Hydro One": "H.TO",
    "Bell Canada": "BCE.TO",
    "Cogeco": "CCA.TO",
    "Quebecor": "QBR-B.TO",
    "Rogers": "RCI-B.TO",
    "Telus": "T.TO"
}

sector_mapping = {
    'ATD.TO': 'consumer_staples',
    'EMP-A.TO': 'consumer_staples',
    'L.TO': 'consumer_staples',
    'MRU.TO': 'consumer_staples',
    'CNR.TO': 'industry_rail',
    'CP.TO': 'industry_rail',
    'ALA.TO': 'utilities',
    'FTS.TO': 'utilities',
    'H.TO': 'utilities',
    'BCE.TO': 'telecom',
    'CCA.TO': 'telecom',
    'QBR-B.TO': 'telecom',
    'RCI-B.TO': 'telecom',
    'T.TO': 'telecom'
}

relevant_attributes = [
    'dividendRate', 'dividendYield', 'beta', 'forwardPE', 'marketCap',
    'profitMargins', 'shortRatio', 'priceToBook', 'freeCashflow', 'ebitdaMargins'
]

os.environ["AWS_ACCESS_KEY_ID"] = "AKIAZXNNZJEPQOQ6SCAT"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

embedder_model_id = "amazon.titan-embed-text-v2:0"

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

client = boto3.client("bedrock-runtime", region_name="us-west-2")


with open(os.path.join('data', 'tabular_data', 'technical_melted.pkl'), 'rb') as f:
    technical_melted = pickle.load(f)

with open(os.path.join('data', 'tabular_data', 'actual_data.pkl'), 'rb') as f:
    actual_data = pickle.load(f)



##### MAIN APP #####
# Initialize the summarization and QA pipelines
summarizer = pipeline("summarization")
qa_pipeline = pipeline("question-answering")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboards", "AI Financial Analyst"])

# Define stock and year options
stocks = [
    "Couche-Tard", "Empire", "Loblaws", "Métro", "Canadian National Railway Company",
    "Canadian Pacific Kansas City", "AltaGas", "Fortis", "Hydro One", "Bell Canada",
    "Cogeco", "Quebecor", "Rogers", "Telus"
]
years = list(range(2018, 2025))

# Define pages as separate functions
def dashboards_page():
    st.title("Dashboards")
    st.write("This is the Dashboards page.")
    
    # Sidebar options for the dashboard page
    selected_stock = st.sidebar.selectbox("Select a stock for dashboard:", [""] + stocks, index=0)

    # Define a function to filter the data based on the selected time period
    def filter_data_by_time_period(data, period):

        end_date = datetime.now()
        # if period == "Last Month":
        #     start_date = end_date - timedelta(days=30)
        if period == "Last 3 Months":
            start_date = end_date - timedelta(days=90)
        elif period == "Last 6 Months":
            start_date = end_date - timedelta(days=180)
        elif period == "Last Year":
            start_date = end_date - timedelta(days=365)
        elif period == "Last 2 Years":
            start_date = end_date - timedelta(days=730)
        elif period == "Last 5 Years":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date
        
        # Convert the date columns to datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.dropna(subset=['Date'])
        else:
            st.error("The data does not contain a 'Date' column.")
            return pd.DataFrame()  # Return an empty DataFrame if 'Date' column is missing

        filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        return filtered_data

    time_period = st.radio(
        "Select a time period:",
        ["Last 3 Months", "Last 6 Months", "Last Year", "Last 2 Years", "Last 5 Years"],
        index=0,
        horizontal=True
    )

    if selected_stock:
        filtered_data = filter_data_by_time_period(technical_melted[technical_melted["Ticker"] == company_tickers[selected_stock]], time_period)
        #filtered_data = filtered_data[filtered_data["Company"] == selected_stock]

        if not filtered_data.empty:
            last_value = filtered_data[filtered_data["Variable"] == "Adj Close"]["Value"].iloc[-1]
            previous_value = filtered_data[filtered_data["Variable"] == "Adj Close"]["Value"].iloc[0]
            difference = last_value - previous_value
            percent_return = (difference / previous_value) * 100

            st.metric(
                label=f"{selected_stock} ({company_tickers[selected_stock]})",
                value=f"${last_value:.2f}",
                delta=f"${difference:.2f} ({percent_return:.2f}%)",
                delta_color="inverse" if difference < 0 else "normal"
            )

            plot_stock_with_indicators(filtered_data, ticker=company_tickers[selected_stock])


            actual_data['Sector'] = actual_data['Ticker'].map(sector_mapping)
            filtered_data = actual_data[actual_data['Attribute'].isin(relevant_attributes)]
            sector_table = display_stock_info(actual_data, filtered_data, company_tickers[selected_stock])
            st.write(sector_table)
            # Streamlit layout for comparison
            st.subheader(f"Compare To: {company_tickers[selected_stock]}")

            # Note: Limit to 4 stocks for the comparison, adjust as needed
            num_stocks = min(len(sector_table), 4)
            cols = st.columns(num_stocks)

            # Populate each column with stock information
            for i, (index, row) in enumerate(sector_table.iterrows()):
                if i >= num_stocks:
                    break
                company_name, ticker = index  # Extract Company and Ticker from the index
                with cols[i]:
                    # Display ticker and company name
                    st.markdown(f"<h4>{ticker}</h4>", unsafe_allow_html=True)
                    st.write(f"{company_name}")

                    # Display required metrics
                    st.write("**Dividend Rate:**", row.get("dividendRate", "N/A"))
                    st.write("**Dividend Yield:**", row.get("dividendYield", "N/A"))
                    st.write("**EBITDA Margins:**", row.get("ebitdaMargins", "N/A"))
                    st.write("**Forward PE:**", row.get("forwardPE", "N/A"))
                    st.write("**Free Cash Flow:**", row.get("freeCashflow", "N/A"))
                    st.write("**Price to Book:**", row.get("priceToBook", "N/A"))
                    st.write("**Profit Margins:**", row.get("profitMargins", "N/A"))
                    st.write("**Short Ratio:**", row.get("shortRatio", "N/A"))
                    st.write("**Sector:**", row.get("sector", "N/A"))

                    # Divider between metrics for clarity
                    st.markdown("<hr>", unsafe_allow_html=True)



        else:
            st.error("No data available for the selected time period.")
    else:
        st.write("Please select a stock to display the dashboard.")



def chatbot_page(stock, year):
    st.title("Chatbot")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", 
                                          "content": f"Hello! You have selected the financial report of {year} for {stock}.\nHow can I assist you with the financial report?"}]

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.chat_message("user").write(chat["content"])
        else:
            st.chat_message("assistant").write(chat["content"])

    # User input for new message
    new_message = st.chat_input("Type your message here...")
    # print("************\n\n\n")
    # print(st.session_state.chat_history)
    # print("************\n\n\n")
    
    if new_message:
        # Append the user message to the chat history
        st.session_state.chat_history.append({"role": "user", "content": new_message})
        st.chat_message("user").write(new_message)  # Display user message on the screen
        
        # Generate AI response
        # ai_response = summarizer(new_message, max_length=150, min_length=30, do_sample=False)
        conversation = format_chat_history(st.session_state.chat_history)

        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n\n")
        # print(conversation)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n\n")

        ai_response, pages = ai_financial_assistant_chatbot(st.session_state.chat_history[-1], 
                                                         client, 
                                                         model_id, 
                                                         embedder_model_id, stock, year)

        # ai_response = query_llm(conversation, client, model_id)
        # print(ai_response)
        response_text = ai_response
        
        # Append the AI response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        shown_text = f"{response_text}\n\n**** The above information were retrived from the following pages in the document : {pages}"
        st.chat_message("assistant").write(shown_text)  # Display AI response on the screen


def financial_analyst_page():
    st.title("AI Financial Analyst")
    st.write("This section provides you with intelligent insights and analysis generated by an financial analyst AI assistant")
    
    # Sidebar options for the financial analyst page
    selected_stock = st.sidebar.selectbox("Select a stock:", [""] + stocks, index=0)
    selected_year = st.sidebar.selectbox("Select a year:", [""] + years, index=0)

    analysis_sections = st.radio(
        "Analysis:",
        ["AI assistant", "Key financial highlights", "Sector-specific", "Sentiment Analysis"],
        index=0,
        horizontal=True
    )
    if analysis_sections == "AI assistant":
        if selected_stock and selected_year:
            chatbot_page(selected_stock, selected_year)
        else:
            st.write("Please select a stock and a year to begin.")

    else:
        if selected_stock and selected_year:
            report, pages = ai_financial_assistant(client, 
                                                model_id, 
                                                embedder_model_id, 
                                                selected_stock, 
                                                selected_year, 
                                                section=analysis_sections)
            
            print(report)

            st.write(f"These information were extracted based on the following pages in the report.\n{pages}")

            st.write(report)



        # PDF file uploader
        uploaded_file = st.file_uploader("Upload the financial report PDF", type="pdf")
        if uploaded_file is not None:
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            report_text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                report_text += page.get_text("text")

            st.text_area("Extracted Report Text", report_text, height=300)

            # Text input for questions
            question = st.text_input("Ask a question about the financial report:")
            if st.button("Get Answer"):
                if report_text and question:
                    answer = qa_pipeline(question=question, context=report_text)
                    st.subheader("Answer")
                    st.write(answer['answer'])
                else:
                    st.error("Please upload a PDF and enter a question.")

# Display the appropriate page based on the selected option
if page == "Dashboards":
    dashboards_page()
elif page == "Chatbot":
    
    pass
elif page == "AI Financial Analyst":
    financial_analyst_page()
    pass