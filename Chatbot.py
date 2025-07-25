import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from PyPDF2 import PdfReader
from docx import Document

# Initialize SentenceTransformer and Groq client
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# For deployment on Streamlit
#api_key = st.secrets['key']
#client = Groq(api_key=api_key)
client = Groq(api_key="use your groq api key")

# Global variables
documents = []
document_embeddings = []

# Function to extract text from PDF with exception handling
def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return None

# Function to extract text from DOCX with exception handling
def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return None

# Function to extract text from TXT with exception handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        return None

# Function to extract text based on file type
def extract_text(file, file_type):
    valid_extensions = ['pdf', 'docx', 'txt']
    if file_type not in valid_extensions:
        raise ValueError(f"Unsupported file format: {file_type}. Please upload a PDF, DOCX, or TXT file.")

    if file_type == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_type == 'docx':
        text = extract_text_from_docx(file)
    elif file_type == 'txt':
        text = extract_text_from_txt(file)
    else:
        text = None

    if text is None:
        raise ValueError(f"Failed to extract text from {file_type.upper()} file. The file may be corrupted or empty.")
    return text

# Function to split long text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Update knowledge base
def update_knowledge_base(text):
    global documents, document_embeddings
    chunks = split_text_into_chunks(text)
    documents.extend(chunks)
    document_embeddings = retriever.encode(documents, convert_to_tensor=True)

# Retrieve relevant context from stored embeddings
def retrieve(query, top_k=1):
    if not documents or len(document_embeddings) == 0:
        return None

    query_embedding = retriever.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, document_embeddings, top_k=top_k)

    if hits and hits[0]:
        corpus_id = hits[0][0]['corpus_id']
        return documents[corpus_id]
    else:
        return None

# Generate response using Groq's LLM
def generate_response(query, context):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # or "mixtral-8x7b"
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{query}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠ Error generating response: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Document Chatbot with Groq", layout="wide", initial_sidebar_state="expanded")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    if st.button("About This Project"):
        st.session_state.show_about = not st.session_state.get('show_about', False)

# Main page
st.title("📄 Document Chatbot")
st.markdown("Upload a PDF, DOCX, or TXT file and ask questions based on its content.")

# About section as an expander
if st.session_state.get('show_about', False):
    with st.expander("About This Project", expanded=True):
        st.markdown("""
        This is a Retrieval-Augmented Generation (RAG) application built to process PDF, DOCX, and TXT files and answer questions based on their content using Groq's language model and SentenceTransformers for semantic search.

        ### Project Team
        - *Zubair Khalil (231204)*
        - *Abdul Rehman Ali (231178)*
        - *Rabia Batool (231142)*

        ### Submission
        Submitted to *Mam Faiza Qamar*.

        ### Technology Stack
        - *Streamlit*: For the web interface
        - *PyPDF2*: For PDF text extraction
        - *python-docx*: For DOCX text extraction
        - *SentenceTransformers*: For semantic text embeddings
        - *Groq*: For generating responses using LLMs
        """)

# Upload file
uploaded_file = st.file_uploader("📤 Upload a file", type=["pdf", "docx", "txt"])
if uploaded_file:
    try:
        with st.spinner("📚 Reading and indexing file..."):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            text = extract_text(uploaded_file, file_extension)
            if text and text.strip():
                update_knowledge_base(text)
                st.success(f"✅ {file_extension.upper()} content processed and indexed.")
            else:
                raise ValueError(f"No extractable text found in the {file_extension.upper()} file. It may be empty or contain only non-text content.")
    except ValueError as ve:
        st.error(f"❌ {str(ve)}")
    except Exception as e:
        st.error(f"❌ Error while processing the file: {str(e)}")

# Ask a question
question = st.text_input("💬 Ask your question here:")
if question:
    with st.spinner("🔎 Retrieving context..."):
        context = retrieve(question)
    if context:
        with st.spinner("🤖 Generating answer..."):
            answer = generate_response(question, context)
            st.markdown(f"*Answer:* {answer}")
    else:
        st.warning("⚠ No relevant context found to answer your question.")
