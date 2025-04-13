import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import re

app = Flask(__name__)

# Load models
MODEL_DIR = 'models'
MAX_LEN = 200

# Load the best traditional model (assuming it's saved as best_traditional_model.joblib)
traditional_model_path = os.path.join(MODEL_DIR, 'logistic_regression.joblib')
traditional_model = joblib.load(traditional_model_path)

# Load LSTM model and tokenizer
lstm_model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
lstm_model = load_model(lstm_model_path)

tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    """Basic text cleaning for traditional model"""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_traditional(text):
    """Make prediction using traditional model"""
    cleaned_text = clean_text(text)
    prediction = traditional_model.predict([cleaned_text])[0]
    probability = traditional_model.predict_proba([cleaned_text])[0]
    
    if prediction == 1:
        sentiment = "Positive"
        prob = probability[1]
    else:
        sentiment = "Negative"
        prob = probability[0]
    
    return sentiment, prob

def predict_lstm(text):
    """Make prediction using LSTM model"""
    # Tokenize and pad sequence
    sequences = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(sequences, maxlen=MAX_LEN)
    
    # Predict
    prediction = lstm_model.predict(padded_seq)[0][0]
    
    if prediction > 0.5:
        sentiment = "Positive"
        prob = float(prediction)
    else:
        sentiment = "Negative"
        prob = 1 - float(prediction)
    
    return sentiment, prob

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Get prediction from models"""
    data = request.get_json()
    text = data['text']
    model_type = data.get('model_type', 'traditional')
    
    if model_type == 'traditional':
        sentiment, probability = predict_traditional(text)
    else:  # deep learning
        sentiment, probability = predict_lstm(text)
    
    # Find important words (simplified approach)
    words = text.lower().split()
    important_words = []
    
    # Very simplistic approach - in production you'd use LIME or SHAP
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'like']
    negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'disappointed', 'poor']
    
    if sentiment == "Positive":
        important_words = [word for word in words if word in positive_words]
    else:
        important_words = [word for word in words if word in negative_words]
    
    return jsonify({
        'sentiment': sentiment,
        'probability': round(probability * 100, 2),
        'highlighted_words': important_words
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a basic HTML template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        textarea {
            width: 100%;
            height: 100px;
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .positive {
            background-color: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
        }
        .negative {
            background-color: rgba(244, 67, 54, 0.2);
            border: 1px solid #F44336;
        }
        .highlight {
            background-color: yellow;
            padding: 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis</h1>
        <p>Enter a text review to analyze its sentiment:</p>
        <textarea id="text-input" placeholder="Enter your text here..."></textarea>
        <div>
            <button onclick="predict('traditional')">Analyze with Traditional ML</button>
            <button onclick="predict('deep')">Analyze with Deep Learning</button>
        </div>
        <div id="result" class="result">
            <h3>Analysis Result</h3>
            <p><strong>Sentiment:</strong> <span id="sentiment"></span></p>
            <p><strong>Confidence:</strong> <span id="confidence"></span>%</p>
            <p><strong>Text with highlights:</strong> <div id="highlighted-text"></div></p>
        </div>
    </div>

    <script>
        function predict(modelType) {
            const text = document.getElementById('text-input').value;
            if (!text) {
                alert('Please enter some text');
                return;
            }
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model_type: modelType
                }),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('sentiment').textContent = data.sentiment;
                document.getElementById('confidence').textContent = data.probability;
                
                // Display highlighted text
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'result ' + (data.sentiment.toLowerCase() === 'positive' ? 'positive' : 'negative');
                
                // Highlight important words
                let textWithHighlights = text;
                data.highlighted_words.forEach(word => {
                    const regex = new RegExp('\\b' + word + '\\b', 'gi');
                    textWithHighlights = textWithHighlights.replace(regex, `<span class="highlight">${word}</span>`);
                });
                
                document.getElementById('highlighted-text').innerHTML = textWithHighlights;
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error analyzing text');
            });
        }
    </script>
</body>
</html>
            ''')
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)