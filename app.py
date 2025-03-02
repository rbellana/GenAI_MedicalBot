import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"

# Load Google Gemini Pro Model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Function to read and process Excel data
def load_excel_data(uploaded_file):
    df = pd.read_excel(uploaded_file)  # Read the uploaded Excel file
    records = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        formatted_text = ", ".join([f"{col}: {val}" for col, val in row_dict.items()])  # Convert row into structured text
        records.append(formatted_text)

    return df, records  # Return dataframe and processed records

# Generate Embeddings & Store in FAISS
def create_faiss_index(records):
    embeddings = [embedding_model.embed_query(text) for text in records]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, records

# Search for Similar Case in FAISS
def find_similar_case(query, index, records):
    query_vector = np.array([embedding_model.embed_query(query)])
    distances, indices = index.search(query_vector, k=1)
    return records[indices[0][0]] if indices[0][0] < len(records) else "No similar case found."

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
st.write("Upload an Excel file with patient data and enter symptoms to find similar cases.")

# File Upload Widget
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    df, patient_records = load_excel_data(uploaded_file)
    faiss_index, processed_data = create_faiss_index(patient_records)
    
    st.success("Excel file processed successfully! Now enter symptoms below.")

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
