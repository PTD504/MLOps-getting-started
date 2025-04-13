import os
import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import re
from pathlib import Path

# Model cho API request
class TextInput(BaseModel):
    text: str
    model_type: str = "traditional"  # "traditional" hoặc "deep"

# Khởi tạo FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API phân tích cảm xúc văn bản sử dụng ML và Deep Learning",
    version="1.0.0"
)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục templates và static nếu chưa có
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Tạo template cho trang web
templates = Jinja2Templates(directory="templates")

# Thiết lập các thư mục tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models
MODEL_DIR = 'models'
MAX_LEN = 200

# Load the traditional model
try:
    traditional_model_path = os.path.join(MODEL_DIR, 'logistic_regression.joblib')
    traditional_model = joblib.load(traditional_model_path)
    print(f"Đã tải mô hình truyền thống từ {traditional_model_path}")
except Exception as e:
    print(f"Không thể tải mô hình truyền thống: {e}")
    traditional_model = None

# Load LSTM model and tokenizer
try:
    lstm_model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Đã tải mô hình LSTM từ {lstm_model_path}")
except Exception as e:
    print(f"Không thể tải mô hình LSTM: {e}")
    lstm_model = None
    tokenizer = None

# Create HTML template file if it doesn't exist
index_html_path = templates_dir / "index.html"
if not index_html_path.exists():
    with open(index_html_path, "w", encoding="utf-8") as f:
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
    if traditional_model is None:
        return "Không có dữ liệu", 0.0
        
    cleaned_text = clean_text(text)
    prediction = traditional_model.predict([cleaned_text])[0]
    
    # Check if model has predict_proba method
    if hasattr(traditional_model, 'predict_proba'):
        probability = traditional_model.predict_proba([cleaned_text])[0]
        
        if prediction == 1:
            sentiment = "Positive"
            prob = probability[1]
        else:
            sentiment = "Negative"
            prob = probability[0]
    else:
        # For models without predict_proba (like LinearSVC)
        sentiment = "Positive" if prediction == 1 else "Negative"
        prob = 0.8  # Default confidence
    
    return sentiment, prob

def predict_lstm(text):
    """Make prediction using LSTM model"""
    if lstm_model is None or tokenizer is None:
        return "Không có dữ liệu", 0.0
        
    # Tokenize and pad sequence
    sequences = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    
    # Predict
    prediction = lstm_model.predict(padded_seq)[0][0]
    
    if prediction > 0.5:
        sentiment = "Positive"
        prob = float(prediction)
    else:
        sentiment = "Negative"
        prob = 1 - float(prediction)
    
    return sentiment, prob

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_data: TextInput):
    """Get prediction from models"""
    text = input_data.text
    model_type = input_data.model_type
    
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
    
    return {
        'sentiment': sentiment,
        'probability': round(probability * 100, 2),
        'highlighted_words': important_words
    }

# Thêm các metadata cho API docs
@app.get("/info")
async def get_info():
    """Hiển thị thông tin về API"""
    return {
        "app_name": "Sentiment Analysis API",
        "version": "1.0.0",
        "models_available": {
            "traditional": traditional_model is not None,
            "lstm": lstm_model is not None
        }
    }