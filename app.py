from flask import Flask, request, jsonify, render_template, send_file
import os
import json
import time
import torch
from pathlib import Path
from TTS.utils.synthesizer import Synthesizer
import fitz  # PyMuPDF
from textblob import TextBlob
import spacy
# import textblob
# textblob.download_corpora()
import nltk
nltk.download('punkt_tab')

from transformers import pipeline, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)

print("Flask app initialized")

# Define paths
base_model_path = Path(r"C:\Users\HP\Desktop\nlp\trump.pth")
config_path = Path(r"C:\Users\HP\Desktop\nlp\models_config.json")
models_json_path = Path(r"C:\Users\HP\Desktop\nlp\models.json")
output_path = Path(r"C:\Users\HP\Desktop\nlp\results")
os.makedirs(output_path, exist_ok=True)

print("Paths defined")

# Load models configuration from JSON file
voices = {}
with open(models_json_path, "r") as f:
    print(f"Loading models from {models_json_path}")
    models = json.load(f)
    for voice in models["voices"]:
        voices[voice["name"]] = voice

print("Models loaded")

# Set CUDA usage based on availability
use_cuda = torch.cuda.is_available()
print(f"CUDA available: {use_cuda}")

# Load NLP models once
print("Loading NLP models")
nlp = spacy.load('en_core_web_sm')
summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
print("NLP models loaded")

# Function to synthesize text to audio and save the output
def synthesize(text: str):
    print("Starting synthesis")
    model_file = base_model_path
    if not model_file.exists():
        print(f"Model file {model_file} does not exist.")
        raise FileNotFoundError(f"Model file {model_file} does not exist.")
    if not config_path.exists():
        print(f"Config file {config_path} does not exist.")
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    
    print(f"Initializing Synthesizer with config path: {config_path} and model file: {model_file}")
    synthesizer = Synthesizer(tts_config_path=config_path, tts_checkpoint=model_file, use_cuda=use_cuda)
    print("Synthesizer initialized")
    
    wav = synthesizer.tts(text)
    output_filename = f"{int(time.time())}_trump.wav"
    output_file_path = output_path / output_filename
    synthesizer.save_wav(wav, output_file_path)
    print(f"Audio saved to {output_file_path}")
    
    return output_file_path

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from PDF: {pdf_path}")
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    print("Text extraction complete")
    return text

# Function to split text into smaller chunks
def split_text_into_chunks(text, max_tokens=1024):
    print("Splitting text into chunks")
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(tokenizer(' '.join(current_chunk))['input_ids']) >= max_tokens:
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    print("Text splitting complete")
    return chunks

# Function to analyze text for persons, places, and emotional words
def analyze_text(text):
    print("Analyzing text")
    doc = nlp(text)
    persons = list(set([ent.text for ent in doc.ents if ent.label_ == 'PERSON']))
    places = list(set([ent.text for ent in doc.ents if ent.label_ in ('GPE', 'LOC')]))
    emotional_words = extract_emotional_words(text)
    print("Text analysis complete")
    return persons, places, emotional_words

def extract_emotional_words(text):
    print("Extracting emotional words")
    emotional_words = []
    emotion_keywords = {
        'happy': ['happy', 'joy', 'elated', 'excited'],
        'sad': ['sad', 'unhappy', 'down', 'depressed'],
        'angry': ['angry', 'mad', 'furious', 'irritated'],
    }
    blob = TextBlob(text)
    for word in blob.words:
        for emotion, keywords in emotion_keywords.items():
            if word.lower() in keywords:
                emotional_words.append((word, emotion))
    print("Emotional words extraction complete")
    return emotional_words

def determine_tone(emotional_words):
    print("Determining tone")
    emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0}
    for _, emotion in emotional_words:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1

    if not any(emotion_counts.values()):
        print("Tone: Neutral")
        return "Neutral"

    tone = max(emotion_counts, key=emotion_counts.get)
    print(f"Tone: {tone.capitalize()}")
    return tone.capitalize()

def summarize_text(text):
    print("Summarizing text")
    chunks = split_text_into_chunks(text)
    summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    combined_summary = ' '.join(summaries)
    print("Text summarization complete")
    return combined_summary

@app.route('/')
def index():
    print("Rendering index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    print("Handling PDF upload")
    if 'pdf' not in request.files:
        print("No PDF file uploaded")
        return jsonify({"error": "No PDF file uploaded"}), 400

    pdf_file = request.files['pdf']
    pdf_path = os.path.join(output_path, pdf_file.filename)
    pdf_file.save(pdf_path)
    print(f"PDF saved to {pdf_path}")

    # Extract, summarize, analyze, and synthesize
    text = extract_text_from_pdf(pdf_path)
    summary = summarize_text(text)
    persons, places, emotional_words = analyze_text(text)
    tone = determine_tone(emotional_words)

    audio_file_path = synthesize(summary)

    print("Rendering result page")
    return render_template(
        'result.html',
        summary=summary,
        persons=persons,
        places=places,
        emotional_words=emotional_words,
        tone=tone,
        audio_file=audio_file_path.name
    )

@app.route('/playback/<audio_file>')
def playback(audio_file):
    print(f"Playing back audio file: {audio_file}")
    return send_file(os.path.join(output_path, audio_file), as_attachment=False)

if __name__ == "__main__":
    print("Starting Flask server")
    app.run(debug=True)