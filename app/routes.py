from flask import render_template, request
from app import app
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model/phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    return text.lower()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  # ✅ Define result before condition
    if request.method == 'POST':
        email_content = request.form['email']
        clean_text = preprocess(email_content)
        vectorized_input = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_input)[0]
        result = 'Phishing Email ⚠️' if prediction == 1 else 'Legitimate ✅'
    return render_template('index.html', result=result)
