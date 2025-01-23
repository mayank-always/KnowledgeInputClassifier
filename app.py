import streamlit as st
import PyPDF2
from transformers import pipeline
import os

# Load a pre-trained classification model
# To be replaced with custom trained model in near future
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define  Streamlit UI
st.title("Knowledge Input Classifier 0.0")
st.write("User can upload a transcript PDF to classify the tag that best fits the discussion category.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extracting text from the PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    # Display extracted text (optional, for user preview)
    st.subheader("Extracted Text Preview")
    st.text_area("Extracted Text", value=text[:1000], height=300)

    # Predict the category
    st.subheader("Classification Result")
    with st.spinner("Classifying..."):
        prediction = classifier(text, truncation=True, max_length=512)
        best_category = prediction[0]['label']

    # Display the predicted category
    st.write(f"Predicted Tag: **{best_category}**")

