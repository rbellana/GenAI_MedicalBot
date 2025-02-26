import os
import faiss
import numpy as np
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader


# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"

# Load Google Gemini Pro Model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

# Load Google Embedding Model for FAISS
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# def load_patient_data():
#     records = []
#     filenames = sorted(os.listdir("data"))  # Read files in order
#     for filename in filenames:
#         with open(f"data/{filename}", "r", encoding="utf-8") as file:
#             text = file.read().strip()
#             records.append(text)
#     return records

# patient_records = load_patient_data()

import os
import re
from langchain.document_loaders import PyPDFLoader

def clean_text(text):
    """Remove unnecessary newlines within sentences while keeping intended paragraph breaks."""
    text = re.sub(r'\n\s*', ' ', text)  # Replace newlines followed by spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
    return text.strip()

def load_patient_data():
    records = []
    filenames = sorted(os.listdir("pdf_data"))  # Read files in order

    for filename in filenames:
        if filename.endswith(".pdf"):  # Ensure only PDFs are read
            loader = PyPDFLoader(f"pdf_data/{filename}")
            pages = loader.load()
            text = " ".join([page.page_content for page in pages])  # Extract text from all pages
            clean_record = clean_text(text)  # Clean text
            records.append(clean_record)

    return records

patient_records = load_patient_data()

# Generate Embeddings & Store in FAISS
def create_faiss_index(patient_records):
    embeddings = [embedding_model.embed_query(text) for text in patient_records]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, patient_records

faiss_index, processed_data = create_faiss_index(patient_records)

# Search for Similar Case in FAISS
def find_similar_case(query, index, patient_records):
    query_vector = np.array([embedding_model.embed_query(query)])
    distances, indices = index.search(query_vector, k=1)
    return patient_records[indices[0][0]] if indices[0][0] < len(patient_records) else "No similar case found."

# Generate Final Diagnosis & Treatment using Gemini Pro
def generate_medical_response(symptoms, retrieved_case):
    prompt = PromptTemplate.from_template(
        """You are an expert medical assistant. A patient has described their symptoms as follows:
        Symptoms: {symptoms}

        Based on a similar past case:
        {retrieved_case}

        Now, provide the final response including:
        1. Diagnosis
        2. Initial Treatment
        3. Follow-up Recommendations"""
    )
    final_prompt = prompt.format(symptoms=symptoms, retrieved_case=retrieved_case)
    response = model.invoke(final_prompt)
    return response.content

# Streamlit UI
st.title("💊 AI-Powered Medical Diagnosis Assistant")

st.write("Enter your symptoms, and we'll find a similar past case and provide treatment suggestions.")

user_input = st.text_area("Describe your symptoms:")

if st.button("Find Diagnosis & Treatment"):
    if user_input:
        similar_case = find_similar_case(user_input, faiss_index, patient_records)
        final_response = generate_medical_response(user_input, similar_case)
        
        st.subheader("🔍 Closest Matching Case")
        st.write(similar_case)
        
        st.subheader("💡 AI-Generated Medical Advice")
        st.write(final_response)
    else:
        st.warning("Please enter symptoms before searching.")