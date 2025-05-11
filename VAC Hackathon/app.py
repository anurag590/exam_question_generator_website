from flask import Flask, render_template, request, send_from_directory
from transformers import T5Tokenizer, T5ForConditionalGeneration
from keybert import KeyBERT
from fpdf import FPDF
import pymupdf as fitz
import os
import unicodedata
import torch

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PDF_FILENAME = "questions.pdf"
PDF_PATH = os.path.join(UPLOAD_FOLDER, PDF_FILENAME)

# Load tokenizer and model
try:
    tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
    model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

kw_model = KeyBERT()

# -------- Utility Functions -------- #

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return "".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def extract_key_concepts(text, top_n=5):
    try:
        return [kw[0] for kw in kw_model.extract_keywords(text, top_n=top_n, stop_words='english')]
    except Exception as e:
        print(f"Error extracting key concepts: {e}")
        return []

def generate_question(concept):
    try:
        input_ids = tokenizer.encode(f"generate question: {concept}", return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = model.generate(input_ids, max_length=64, do_sample=True, temperature=0.7)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Error generating question for '{concept}': {e}")
        return "Error generating question"

def clean_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def create_pdf(college_name, num_questions, total_marks, difficulty, questions, path):
    try:
        pdf = FPDF()
        pdf.add_page()

        # Title Header
        pdf.set_font("Arial", size=16, style='B')
        pdf.cell(200, 10, txt=f"{college_name} - Question Paper", ln=True, align='C')

        # Line after title
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
        pdf.ln(8)

        # Metadata
        pdf.set_font("Arial", size=12)
        pdf.cell(100, 10, f"Difficulty Level: {difficulty}", ln=True)
        pdf.cell(100, 10, f"Total Marks: {total_marks}", ln=True)
        pdf.cell(100, 10, f"Number of Questions: {num_questions}", ln=True)

        # Line after metadata
        pdf.ln(2)
        pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
        pdf.ln(10)

        # Questions Section
        for i, q in enumerate(questions[:num_questions], 1):
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Q{i}: {clean_text(q)}")
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # Line between questions
            pdf.ln(4)

        pdf.output(path)
    except Exception as e:
        print(f"Error creating PDF: {e}")

# -------- Routes -------- #

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload-form')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files or request.files['pdf_file'].filename == '':
        return "No file uploaded", 400

    file = request.files['pdf_file']
    college_name = request.form.get('college_name')
    num_questions = int(request.form.get('num_questions'))
    total_marks = int(request.form.get('total_marks'))
    difficulty = request.form.get('difficulty_level')

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    if not text:
        return "Failed to extract text", 500

    # Extract more key concepts than required
    concepts = extract_key_concepts(text, top_n=num_questions * 2)
    if not concepts:
        return "No key concepts found", 500

    # Generate questions
    questions = [generate_question(c) for c in concepts]

    # Fill to required number of questions
    if len(questions) < num_questions:
        while len(questions) < num_questions:
            questions.append(questions[len(questions) % len(concepts)])
    else:
        questions = questions[:num_questions]

    create_pdf(college_name, num_questions, total_marks, difficulty, questions, PDF_PATH)

    return render_template('download.html')

@app.route('/download')
def download():
    if os.path.exists(PDF_PATH):
        return send_from_directory(UPLOAD_FOLDER, PDF_FILENAME, as_attachment=True)
    return "No file found for download", 404

# -------- Run App -------- #

if __name__ == "__main__":
    app.run(debug=True)
