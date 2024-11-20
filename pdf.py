import streamlit as st
import fitz  # PyMuPDF
import cohere
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Cohere API
API_KEY = "opKHtxN5LG2MLT9RGj2vanYfhTq4Y7TCS2qGkjPn"  # Replace with your Cohere API key
co = cohere.Client(API_KEY)

# Function to extract text from the uploaded PDF file
def extract_text_from_pdf(pdf_filename):
    # Check if file exists
    if not os.path.isfile(pdf_filename):
        raise ValueError(f"File '{pdf_filename}' not found in the current directory.")
    
    try:
        # Open the PDF file with fitz (PyMuPDF)
        document = fitz.open(pdf_filename)  # Directly open the file without using BytesIO
    except Exception as e:
        raise ValueError(f"Error opening PDF file: {e}")
    
    # Extract text from all pages
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to generate response using Cohere API
def generate_response(prompt, context, max_tokens=300, temperature=0.7):
    try:
        response = co.generate(
            model="command-xlarge",  # Use a valid Cohere model ID
            prompt=f"Context: {context}\n\nQuestion: {prompt}\nAnswer:",
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=["\n"],
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# Function to retrieve relevant text from the PDF based on the query
def retrieve_relevant_text(query, pdf_text, top_n=3):
    # Preprocess PDF text and query
    paragraphs = pdf_text.split('\n')
    
    # Using TF-IDF to retrieve the most relevant paragraphs
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(paragraphs + [query])
    cosine_similarities = np.dot(tfidf_matrix[:-1], tfidf_matrix[-1].T).toarray().flatten()
    
    # Sort by relevance (cosine similarity) and return top N relevant paragraphs
    relevant_paragraphs = [paragraphs[i] for i in cosine_similarities.argsort()[-top_n:]]
    return "\n".join(relevant_paragraphs)

# Streamlit app UI
st.title("RAG PDF with Cohere API")
st.write("Provide a PDF filename and ask questions based on its content!")

# User input for the PDF file name
pdf_filename = st.text_input("Enter the PDF file name (with extension):", "What is a Knight.pdf")

# Once the file name is provided, extract the text and allow user to ask a question
if pdf_filename:
    # Extract text from the PDF
    try:
        pdf_text = extract_text_from_pdf(pdf_filename)
        st.write(f"PDF text extracted successfully from '{pdf_filename}'!")
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        pdf_text = ""

    # Ask user for a query
    query = st.text_input("Ask a question based on the PDF content:")

    if query and pdf_text:
        # Retrieve relevant text from the PDF based on the user's query
        relevant_text = retrieve_relevant_text(query, pdf_text)
        st.write("Relevant Text Retrieved from PDF:")
        st.write(relevant_text)

        # Generate a response based on the context from the PDF and the user's query
        response = generate_response(query, relevant_text)
        st.subheader("Generated Response:")
        st.write(response)

