# Document Chatbot with Groq

A Streamlit-based chatbot that lets you upload PDF, DOCX, or TXT files and ask questions about their content. It uses SentenceTransformers for context retrieval and Groq’s LLM for generating intelligent answers. Ideal for document Q&A and research support.

## Features

- Upload and process PDF, DOCX, and TXT files
- Extract text and chunk it into semantically meaningful segments
- Semantic search using SentenceTransformers
- Natural language answer generation using Groq LLMs (LLaMA 3 or Mixtral)
- Simple and clean Streamlit UI
- Error handling for corrupted or unsupported files

## Tech Stack

- Python
- Streamlit
- SentenceTransformers
- Groq API (LLaMA 3 or Mixtral models)
- PyPDF2 (for PDF)
- python-docx (for DOCX)

## Installation

Create a virtual environment (optional but recommended):


    python -m venv env
    source env/bin/activate  # or .\env\Scripts\activate on Windows

## Install Dependencies

    pip install -r requirements.txt

## Running the App

    streamlit run chatbot.py

