import os
from fpdf import FPDF

# Sample transcripts data (this should mirror your actual data structure)
transcripts = [
    {
        "id": "elem_001",
        "category": "Elementary",
        "transcript": "Call Date: 2025-04-01\nDuration: 00:14:30\nParticipants: Caller (Parent: Mrs. Henderson), Agent (Admissions Officer: Mr. Baker)\n\n[Transcript Start]\nMr. Baker: Good morning... [Transcript End]"
    },
    # ... include remaining transcript entries here ...
]

output_dir = "pdf_transcripts"
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "..."
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text

class TranscriptPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Transcript", ln=True, align="C")
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

for transcript in transcripts:
    pdf = TranscriptPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)
    
    cleaned_text = clean_text(transcript["transcript"])
    
    for line in cleaned_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    
    filename = os.path.join(output_dir, f"{transcript['id']}.pdf")
    pdf.output(filename)
    print(f"Created {filename}")

print("All PDFs have been created in the 'pdf_transcripts' folder.")
