# assistant.py

import os
import boto3
from langchain_community.document_loaders import PyPDFDirectoryLoader
from utils import generate_database, get_relevant_context, query_llm, assemble_analysis_prompt, assemble_rag_query
import warnings
warnings.filterwarnings("ignore")

# Define paths
DATABASE_PATH = os.environ.get("DATABASE_PATH", "data/database")
DATA_PATH = os.environ.get("DATA_PATH", "data/docs")
TEMPLATE_PATH = os.environ.get("TEMPLATE_PATH", "data/templates")

# Initialize client and model ID
model_id = "anthropic.claude-3-haiku-20240307-v1:0"
client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Define the prompt template for the assistant
# PROMPT_TEMPLATE = """
#     <meta>
#         current year: 2024
#         role: financial analyst
#         language: English
#         expertise: finance, annual reports, financial performance, sector-specific insights
#         response style: factual, analytical, detailed
#         response size: very detailed, comprehensive, clear
#     </meta>

#     Important: 
#     - Do not mention being an AI or assistant.
#     - Answer based on provided context, if the question is finance-related, offer financial insights, indicators, and trends.

#     Question: {question}
#     Context: {context}
# """

# Generate the document database
if not os.path.exists(DATABASE_PATH):
    os.makedirs(DATABASE_PATH)
    print("Generating the document database...")
    documents = generate_database(DATA_PATH)
    print('Database generated.')
    
# Define a sample question for the assistant
# question = "How is Bell Canada's financial health in 2021 compared to 2020?"

# Retrieve relevant context for the question
conversation, relevant_results = get_relevant_context(template_dir = os.path.join(TEMPLATE_PATH, "analysis_basic_indicators.toml"), k=5)

# print(f"Retrieved {len(relevant_results)} relevant documents.")
# print("Database Path:", DATABASE_PATH)
# print("Data Path:", DATA_PATH)
# loader = PyPDFDirectoryLoader(DATA_PATH)
# documents = loader.load()
# print(f"Loaded {len(documents)} documents.")
# print("Document Metadata:", [doc.metadata for doc in documents])
# Query the model with the formatted conversation
response_text = query_llm(conversation, client, model_id)
sources = [doc.metadata.get("id", None) for doc, _score in relevant_results]
formatted_response = f"Assistant: {response_text}\nSources: {sources}"

# Print the assistant's response and sources
print(formatted_response)