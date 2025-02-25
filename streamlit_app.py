import os
import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import docx  # For DOCX extraction
from transformers import pipeline

# Step 1: Ensure the trained model exists
MODEL_PATH = "./trained_model"
if not os.path.exists(MODEL_PATH):
    st.error("🚨 Trained model not found! Please run `train_model.py` first.")
    st.stop()

# Step 2: Load the trained model (with caching)
@st.cache_resource
def load_model():
    return pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)

classifier = load_model()

# Function to extract text from a DOCX file
def extract_text_from_docx(file):
    doc = docx.Document(file)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join([page.get_text("text") for page in doc])

# Step 3: Define the Streamlit UI
st.title("Knowledge Input Classifier 3.0")
st.write("Upload a transcript file. You can upload either a PDF transcript or a DOCX transcript using the boxes below. This model currently classifies educational tags only.")

# Step 4: Provide separate uploaders for PDF and DOCX files
pdf_file = st.file_uploader("📂 Upload PDF Transcript", type=["pdf"])
docx_file = st.file_uploader("📂 Upload DOCX Transcript", type=["docx"])

text = None  # variable to hold extracted text

# Step 5: Extract text based on which file is uploaded
if pdf_file is not None:
    st.write("PDF file uploaded:", pdf_file.name)
    text = extract_text_from_pdf(pdf_file)
elif docx_file is not None:
    st.write("DOCX file uploaded:", docx_file.name)
    text = extract_text_from_docx(docx_file)

# Step 6: If text was extracted, show a preview and classify
if text:
    st.subheader("🔍 Extracted Text Preview")
    st.text_area("Extracted Text", value=text[:2000], height=300)

    st.subheader("🧠 Classification Result")
    with st.spinner("🤖 Classifying..."):
        prediction = classifier(text, truncation=True, max_length=512)
        # Expect label to be like "LABEL_0", so extract the number
        label_str = prediction[0]['label']
        try:
            best_category = int(label_str.split('_')[-1])
        except Exception:
            best_category = int(label_str)
    
    # Step 7: Map the numeric label to a human-readable category name
    categories = ["Elementary Education Experience", "Higher Education Experience", "Lifelong Learning"]
    st.write(f"🎯 **Predicted Category:** {categories[best_category]}")
