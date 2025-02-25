# Knowledge Input Transcript Classifier

This repository contains a text classifier designed to categorize call transcripts into three categories: Elementary, Higher Education, and Lifelong Learning. The solution integrates several components:

- **Model Training:** Fine-tune a Hugging Face transformer (e.g., DistilBERT) using custom transcript data.
- **Web Application:** A Streamlit app that accepts transcript uploads (PDF and DOCX) and displays classification results.
- **PDF Generation:** A utility script to convert transcript data into PDF files.

## Components

1. **train_model.py:**  
   Loads a JSON dataset of transcripts, tokenizes the text, fine-tunes a DistilBERT model for classification, and saves the trained model and tokenizer.

2. **streamlit_app.py:**  
   Provides a user interface for uploading transcript files (PDF and DOCX). It extracts text, passes it to the classifier, and displays the predicted category.

3. **generate_pdfs.py:**  
   Generates PDF files from transcript data using the FPDF library (with text cleaning to handle Unicode issues).
