# RAG pipeline for RAG applications in healthcare

## System architecture

![RAG system architecture](https://raw.githubusercontent.com/your-username/your-repo/main/images/system.png)

## Install dependencies

pip install -r requirements.txt

## Install and configure Ollama with selected base model

on Mac OS : brew install ollama Start Ollama : ollama run llama3

## Server and UI setup

Command to start FAST API server <br>

uvicorn app:app --reload --host 0.0.0.0 --port 8000 <br>

And Streamlit UI <br>

streamlit run ui_streamlit.p <br>

# DEVELOP the RAG PIPELINE

## Build document ingestion pipeline

Purpose: Convert raw PDFs or .txt files to cleaned, structured text <br>

python3 $BASEDIR/main.py step01_ingest --input_filename all

## Create text chunking and embedding system

Purpose: Break cleaned text into chunks Generate embeddings using sentence transformers <br>

python3 $BASEDIR/main.py step02_generate_embeddings --input_filename all

## Implement vector database operations

Purpose: Store and index document embeddings Enable fast similarity search <br>

python3 $BASEDIR/main.py step03_store_vectors --input_filename all

## Develop prompt construction logic

Purpose: Dynamically build prompts based on query + retrieved context (for RAG) Fallback to baseline LLM if use_rag=False <br>

python3 $BASEDIR/main.py step05_generate_response --query_args "$QUERY" --use_rag

## Test the pipeline in CLI

Example query : python3 $BASEDIR/main.py step05_generate_response --query_args "How is rag used in decision support" --use_rag

## Running test files

PYTHONPATH=. pytest tests/"file-name" -s
