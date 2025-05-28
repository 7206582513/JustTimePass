# app.py – SmartDoc Scholar Full Web App with OCR Fallback

from flask import Flask, request, render_template, send_file, jsonify, session
import os
import fitz  # PyMuPDF
import spacy
from gtts import gTTS
import requests
from datetime import datetime
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract
import json
import re
from werkzeug.utils import secure_filename

# === CONFIG ===
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

app = Flask(__name__)
app.secret_key = "smartdoc_secret"
UPLOAD_FOLDER = 'uploads'
SUMMARY_AUDIO = 'static/summary.mp3'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)
nlp = spacy.load("en_core_web_sm")

# === DATABASE SETUP ===
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, query TEXT, response TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# === GROQ API ===
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"

def groq_chat(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    res_json = response.json()

    if "choices" not in res_json:
        print("❌ Groq API Error:", res_json)
        return "❌ LLM Error: " + res_json.get("error", {}).get("message", "Unknown issue")

    return res_json["choices"][0]["message"]["content"]

# === PDF TO TEXT ===
def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    return "\n".join([page.get_text() for page in doc])

# === OCR FALLBACK ===
def extract_text_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print("OCR Extraction Error:", e)
        return ""

# === NLP BULLET SUMMARY ===
def summarize_text(text):
    prompt = f"Summarize this educational note into 5-7 key bullet points:\n\n{text[:3000]}"
    return groq_chat(prompt)

# === AUDIO ===
def generate_audio_summary(text):
    try:
        from gtts import gTTS
        tts = gTTS(text)
        tts.save(SUMMARY_AUDIO)
    except Exception as e:
        print("⚠️ gTTS failed. Using offline pyttsx3 fallback. Error:", e)
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, SUMMARY_AUDIO)
        engine.runAndWait()


# === QUIZ GENERATION ===
def generate_quiz_questions(text):
    prompt = f"""
You are an expert quiz generator.

Generate 5 high-quality multiple-choice questions based on the following educational content.
Each question should have:
- 'question': a clear, informative string
- 'options': a list of 4 full answer strings (not labeled A/B/C)
- 'answer': the full correct answer string (must match one of the options exactly)

Output should be a JSON array only. No markdown, labels, or explanations.

Content:
{text[:2500]}
"""
    response = groq_chat(prompt)
    print("📨 LLM raw response:\n", response)

    try:
        json_match = re.search(r"\[\s*{.*?}\s*\]", response, re.DOTALL)
        if json_match:
            quiz_json = json_match.group(0)
            quiz = json.loads(quiz_json)

            for q in quiz:
                answer = q["answer"]
                if isinstance(answer, str) and answer.strip().isdigit():
                    q["answer"] = q["options"][int(answer.strip())]
                elif isinstance(answer, str) and answer.strip().upper() in ["A", "B", "C", "D"]:
                    idx = ["A", "B", "C", "D"].index(answer.strip().upper())
                    q["answer"] = q["options"][idx]

            print("✅ Fixed and parsed quiz:", quiz)
            return quiz

        raise ValueError("No valid JSON array found.")
    except Exception as e:
        print("❌ Quiz JSON parsing error:", e)
        return []

# === VECTOR STORE ===
def build_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    docs = splitter.create_documents([text])
    if not docs:
        raise ValueError("❌ Unable to generate document chunks from text.")
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def answer_query(query, vs):
    docs = vs.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    return groq_chat(f"Answer using this context:\n{context}\n\nQ: {query}")


# === TEXT SIMPLIFIER ROUTE ===
@app.route("/simplify", methods=["POST"])
def simplify():
    text = request.form.get("text", "")
    if not text.strip():
        return jsonify({"simplified": "⚠️ No text provided."})

    prompt = "Simplify the following text using very easy and short sentences suitable for ADHD/Dyslexic students:\n\n" + text
    simplified = groq_chat(prompt)
    return jsonify({"simplified": simplified})


# === FLASHCARD GENERATOR ROUTE ===
@app.route("/flashcards", methods=["POST"])
def flashcards():
    text = request.form.get("text", "")
    if not text.strip():
        return jsonify({"cards": "⚠️ No input provided."})

    prompt = f"""
Generate 5 flashcards from the following content.

Each flashcard must include:
- question: A short question string.
- answer: A simple, correct answer string.

Output as a JSON list. No explanations or markdown.

Content:
{text}
"""
    response = groq_chat(prompt)

    try:
        cards = json.loads(response)
        formatted = "\n".join([f"Q: {card['question']}\nA: {card['answer']}" for card in cards])
        return jsonify({"cards": formatted})
    except Exception as e:
        print("⚠️ Flashcard JSON parsing failed:", e)
        return jsonify({"cards": response})


# === HISTORY ===
def save_history(user, query, response):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("INSERT INTO history (user, query, response, timestamp) VALUES (?, ?, ?, ?)",
              (user, query, response, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_history(user):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("SELECT query, response, timestamp FROM history WHERE user = ? ORDER BY id DESC", (user,))
    data = c.fetchall()
    conn.close()
    return data

# === ROUTES ===
@app.route("/", methods=["GET", "POST"])
def index():
    summary = questions = ""
    if request.method == "POST":
        file = request.files['pdf']
        if not file or file.filename == "":
            return "❌ No PDF uploaded!"

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        session['user'] = request.remote_addr

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            print("Trying OCR fallback...")
            text = extract_text_with_ocr(file_path)
            if not text.strip():
                return "❌ No readable text found even after OCR."

        print("Text Extracted:", text[:500])
        summary = summarize_text(text)
        generate_audio_summary(summary)
        questions = generate_quiz_questions(text)

        global vectorstore
        vectorstore = build_vector_store(text)

    history = get_history(session.get('user', 'guest'))
    return render_template("index.html", summary=summary, questions=questions, history=history)

@app.route("/chat", methods=["POST"])
def chat():
    query = request.form['query']
    user = session.get('user', 'guest')
    response = answer_query(query, vectorstore)
    save_history(user, query, response)
    return jsonify({"response": response})

@app.route("/audio")
def audio():
    return send_file(SUMMARY_AUDIO)

@app.route("/download_history")
def download_history():
    user = session.get('user', 'guest')
    history = get_history(user)

    filename = f"chat_history_{user}.txt"
    filepath = os.path.join("static", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for q, a, t in history:
            f.write(f"Q: {q}\nA: {a}\nTime: {t}\n\n")

    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )