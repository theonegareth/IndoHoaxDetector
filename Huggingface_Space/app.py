import gradio as gr
import joblib
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# --- 1. One-time Setup ---
# Download NLTK stopwords (Hugging Face Spaces will run this on build)
print("Downloading NLTK data...")
nltk.download('stopwords')
print("NLTK download complete.")

# --- 2. Load Models & Preprocessing Tools ---
# Load all tools into memory once when the app starts
print("Loading model and vectorizer...")
try:
    model = joblib.load('logreg_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer file not found.")
    model, vectorizer = None, None

print("Initializing Indonesian stemmer...")
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
print("Stemmer ready.")

# --- 3. Preprocessing Function (MUST be identical to training) ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            cleaned_tokens.append(stemmed_word)
    return ' '.join(cleaned_tokens)

# --- 4. Main Prediction Function (for Gradio) ---
def predict_news(text):
    if model is None or vectorizer is None:
        return {"Error": 1.0}

    # 1. Preprocess the input text
    cleaned_text = preprocess_text(text)
    
    # 2. Vectorize the cleaned text
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # 3. Get prediction probabilities
    probability = model.predict_proba(vectorized_text)
    
    # 4. Format the output for Gradio's Label component
    # model.classes_ typically returns [0, 1]
    # probability[0] gives probabilities for [class_0, class_1]
    confidences = probability[0]
    
    # Create a dictionary mapping label to confidence
    # Assuming 0 = FAKTA and 1 = HOAX
    # You can check model.classes_ to be sure
    labels = ["FAKTA", "HOAX"]
    output_dict = {labels[i]: float(confidences[i]) for i in range(len(labels))}
    
    return output_dict

# --- 5. Create and Launch the Gradio Interface ---
title = "Detektor Berita Hoax Indonesia"
description = """
Model ini dilatih untuk mendeteksi berita politik Hoax dan Fakta dalam Bahasa Indonesia.
Masukkan judul dan isi berita ke dalam kotak teks di bawah untuk melihat prediksinya.
Model ini menggunakan Logistic Regression dengan TF-IDF.
"""
demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=10, 
        placeholder="Masukkan teks berita (judul dan isi) untuk dianalisis...", 
        label="Teks Berita"
    ),
    outputs=gr.Label(
        num_top_classes=2, 
        label="Prediksi"
    ),
    title=title,
    description=description,
    examples=[
        ["Presiden umumkan libur panjang baru untuk seluruh warga"],
        ["Beredar foto ular raksasa di Kalimantan hasil editan"]
    ]
)

# Launch the app
demo.launch()

