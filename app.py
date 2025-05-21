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
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from pathlib import Path

# Tạo lớp request để nhận văn bản từ người dùng
class TextInput(BaseModel):
    text: str

MAX_LEN = 200  # Số lượng từ tối đa (padding length)

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    """Tiền xử lý văn bản: Tokenization và padding"""
    if tokenizer is not None:
        sequences = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(sequences, maxlen=MAX_LEN)
        return padded_seq
    else:
        # Nếu không có tokenizer, bạn có thể dùng phương pháp đơn giản hơn hoặc chỉ trả về văn bản gốc
        return [text]

# Khởi tạo FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API phân tích cảm xúc văn bản sử dụng mô hình Deep Learning duy nhất",
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

# Load mô hình đã huấn luyện (converted_model.h5)
MODEL_DIR = 'models'
MAX_LEN = 200

model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
model = tf.keras.models.load_model(model_path)
print(f"Đã tải mô hình từ {model_path}")

# Nếu mô hình này cần tokenizer, bạn có thể load tương tự như sau, hoặc dùng tokenizer tích hợp:
tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
tokenizer = None
try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Đã tải tokenizer từ {tokenizer_path}")
except Exception as e:
    print(f"Không tìm thấy hoặc không thể tải tokenizer: {e}")

def clean_text(text):
    """Basic text cleaning"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_deep(text):
    """Dự đoán sentiment sử dụng lstm_model.h5"""
    if tokenizer is None:
        # Nếu không có tokenizer, xử lý text đơn giản (có thể gây ảnh hưởng dự đoán)
        sequences = [[ord(c) for c in text.lower()]]  # Ví dụ chuyển text thành số (nên thay đổi tùy model)
        padded_seq = pad_sequences(sequences, maxlen=MAX_LEN)
    else:
        sequences = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(sequences, maxlen=MAX_LEN)
    
    prediction = model.predict(padded_seq)[0][0]
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
    """Get prediction from the deep learning model"""
    text = input_data.text
    sentiment, probability = predict_deep(text)
    
    words = text.lower().split()
    important_words = []
    
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
        "model_loaded": True
    }
