# POLY_DATATHON_2024: AI Financial Analyst - TEAM #16

The application will be accessible at : http://34.216.224.65:8501

## Overview

POLY_DATATHON_2024 is an AI-powered financial analyst tool that provides intelligent insights into financial data. The project is divided into two main sections:

1. *Dashboard*: Visualizes key financial indicators for selected companies and sectors.
2. *AI Financial Assistant*: An interactive assistant that allows users to query financial reports and obtain contextual insights on demand.

This application is built with Streamlit, leveraging NLP and machine learning models to process and analyze financial data from various sources.

## Project Structure
```bash
POLY_DATATHON_2024
│
├── .streamlit                # Streamlit configuration files
│   └── config.toml
│
├── data                      # Data storage
│   ├── doc_store             # Document storage for financial reports
│   ├── tabular_data          # Tabular data for financial analysis
│   │   ├── doc_store.zip
│   │   └── metadata.csv
│
├── materials                 # Project materials and documentation
│
├── src                       # Source code
│   ├── dbcon_test.py         # Database connection test file
│   ├── functions.py          # Utility functions
│   ├── loader_structured_company.py # Data loader for structured company data
│   ├── structured_data.ipynb # Jupyter notebook for data structuring
│
├── templates                 # TOML templates for various analyses
│   ├── analysis_basic_indicators.toml
│   ├── analysis_press_release.toml
│   ├── analysis_sector.toml
│   ├── analysis_sentiment.toml
│   └── analysis_table_of_content.toml
│
├── static                    # Static files (images, CSS, etc.)
│
├── venv                      # Virtual environment
│   └── .env                  # Environment variables
│
├── app.py                    # Main application entry point
├── assistant.py              # AI financial assistant logic
├── dashboards.py             # Dashboard plotting functions
├── fetch_data.py             # Data fetching logic
├── README.md                 # Project documentation
├── requirements.txt          # Required Python packages
└── utils.py                  # Utility functions
```

## Getting Started

### Prerequisites

1. *Python 3.11+*: Make sure Python is installed.
2. *Virtual Environment*: It is recommended to run this project within a virtual environment.
3. *AWS Credentials*: Set up your AWS access credentials to enable data fetching and NLP model interactions.

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/POLY_DATATHON_2024.git
    cd POLY_DATATHON_2024
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Add your AWS credentials:
      
    ```bash
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key
    export AWS_DEFAULT_REGION=your_region
    ```

### Running the Application

To start the application, use the following command:

```bash
streamlit run app.py
```

Application Sections

	1.	Dashboards:
        •	Visualizes historical stock prices and financial indicators for selected companies.
        •	Use the sidebar to choose a company and time range for data display.
        •	Displays metrics such as dividend rate, EBITDA margins, price-to-book ratio, and more.
	2.	AI Financial Analyst:
        •	Upload a financial report PDF to analyze and interact with the AI assistant. (In future release)
        •	The assistant provides summaries, answers to questions, and sector-specific insights.
        •	Choose a company and year to get relevant financial highlights and…

