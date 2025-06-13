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
import time # For inference time
import logging # For logging

# Prometheus integration
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Histogram # For custom metrics

# --- Logging Setup ---
# Configure logging to output to stdout, which Docker can capture
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()] # Output to console
)
logger = logging.getLogger(__name__)

# --- Prometheus Custom Metrics ---
# You can define these globally
MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Time taken for model inference',
    ['model_name'] # You can add labels like model_version if needed
)
MODEL_CONFIDENCE_SCORE = Gauge(
    'model_confidence_score',
    'Confidence score of the last prediction',
    ['model_name', 'sentiment_label']
)

# Tạo lớp request để nhận văn bản từ người dùng
class TextInput(BaseModel):
    text: str

MAX_LEN = 200  #  (padding length)

# Hàm tiền xử lý văn bản (no changes needed from original for this part)
def preprocess_text(text):
    """Tiền xử lý văn bản: Tokenization và padding"""
    if tokenizer is not None:
        sequences = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(sequences, maxlen=MAX_LEN)
        return padded_seq
    else:
        # Nếu không có tokenizer, bạn có thể dùng phương pháp đơn giản hơn hoặc chỉ trả về văn bản gốc
        logger.warning("Tokenizer is None, using fallback preprocessing.")
        return [text]

# Khởi tạo FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for SA using Deep Learning model",
    version="1.0.0"
)

# Add prometheus instrumentator AFTER app initialization
# This will expose /metrics endpoint
Instrumentator().instrument(app).expose(app)
logger.info("Prometheus Instrumentator configured.")

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

# Create a dummy index.html for testing
if not (templates_dir / "index.html").exists():
    with open(templates_dir / "index.html", "w") as f:
        f.write("<h1>Sentiment Analysis API</h1><p>Send POST requests to /predict</p>")
logger.info("Templates and static directories checked/created.")

# Tạo template cho trang web
templates = Jinja2Templates(directory="templates")

# Thiết lập các thư mục tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load mô hình đã huấn luyện
MODEL_DIR = 'models'
# Ensure models directory exists for the script to run even without actual models for now
Path(MODEL_DIR).mkdir(exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
model = None
try:
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Đã tải mô hình từ {model_path}")
except Exception as e:
    logger.error(f"Không thể tải mô hình từ {model_path}: {e}. API sẽ không hoạt động đúng.")
    # You might want to exit or handle this more gracefully if the model is critical

tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
tokenizer = None
try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info(f"Loading tokenizer from {tokenizer_path}")
except Exception as e:
    logger.warning(f"Mot found tokenizer. ERROR: {e}.")


def clean_text(text):
    """Basic text cleaning"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_deep(text):
    """Dự đoán sentiment sử dụng lstm_model.h5"""
    if model is None:
        logger.error("Mô hình chưa được tải. Không thể dự đoán.")
        # Return a default or raise an error
        return "Error", 0.0

    start_time = time.perf_counter() # More precise than time.time() for duration

    # Clean the text before tokenization
    cleaned_text = clean_text(text)
    logger.info(f"Input text for prediction: '{text}', Cleaned: '{cleaned_text}'")

    if tokenizer is None:
        logger.warning("Tokenizer is None. Using very basic character-level sequence for prediction.")
        # This fallback is unlikely to work well with a real LSTM model trained on word tokens
        sequences = [[ord(c) for c in cleaned_text]]
        padded_seq = pad_sequences(sequences, maxlen=MAX_LEN)
    else:
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded_seq = pad_sequences(sequences, maxlen=MAX_LEN)

    try:
        prediction_value = model.predict(padded_seq)[0][0]
    except Exception as e:
        logger.error(f"Lỗi trong quá trình model.predict: {e}")
        # Potentially re-raise or return an error indicator
        return "Error", 0.0

    inference_time = time.perf_counter() - start_time
    MODEL_INFERENCE_TIME.labels(model_name='lstm_model').observe(inference_time)
    logger.info(f"Model inference time: {inference_time:.4f} seconds")

    if prediction_value > 0.5:
        sentiment = "Positive"
        prob = float(prediction_value)
    else:
        sentiment = "Negative"
        prob = 1 - float(prediction_value) # For negative, probability is 1 - raw_prediction

    MODEL_CONFIDENCE_SCORE.labels(model_name='lstm_model', sentiment_label=sentiment.lower()).set(prob)
    logger.info(f"Dự đoán: Sentiment={sentiment}, Probability={prob:.4f}")

    return sentiment, prob

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the home page"""
    logger.info("GET / - Trang chủ được truy cập")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_data: TextInput):
    """Get prediction from the deep learning model"""
    text = input_data.text
    logger.info(f"POST /predict - Nhận văn bản: '{text}'")

    if not model: # Check if model is loaded
        logger.error("POST /predict - Yêu cầu dự đoán nhưng mô hình không được tải.")
        return {"error": "Model not loaded", "sentiment": "N/A", "probability": 0.0, "highlighted_words": []}

    sentiment, probability = predict_deep(text)

    # If predict_deep returned an error
    if sentiment == "Error":
        return {"error": "Prediction failed", "sentiment": "N/A", "probability": 0.0, "highlighted_words": []}

    words = text.lower().split() # Use original text for word highlighting
    important_words = []

    # Simplified word highlighting for demo
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'like', 'happy', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'disappointed', 'poor', 'sad', 'angry']

    if sentiment == "Positive":
        important_words = [word for word in words if word in positive_words]
    else: # Negative or any other category
        important_words = [word for word in words if word in negative_words]

    logger.info(f"POST /predict - Kết quả: Sentiment={sentiment}, Probability={probability*100:.2f}%, Highlighted={important_words}")
    return {
        'sentiment': sentiment,
        'probability': round(probability * 100, 2),
        'highlighted_words': important_words
    }

@app.get("/info")
async def get_info():
    """Hiển thị thông tin về API"""
    logger.info("GET /info - Thông tin API được truy vấn")
    return {
        "app_name": "Sentiment Analysis API",
        "version": "1.0.0",
        "model_loaded": model is not None, # Check actual model status
        "tokenizer_loaded": tokenizer is not None
    }

# Example of how to simulate an error for testing
@app.get("/error_test")
async def error_test():
    logger.error("GET /error_test - This an error experience!")
    raise ValueError("Thia is an experience for testing logging and error rate.")

if __name__ == "__main__":
    import uvicorn
    # Make sure models and tokenizer exist for local run, or handle their absence
    # For example, create dummy files if they don't exist for dev purposes
    if not Path(model_path).exists():
        logger.warning(f"Dummy model creation as {model_path} not found. THIS IS NOT A REAL MODEL.")
        # Create a very simple dummy model if not present
        dummy_input = tf.keras.Input(shape=(MAX_LEN,))
        dummy_output = tf.keras.layers.Dense(1, activation='sigmoid')(dummy_input)
        dummy_model = tf.keras.Model(inputs=dummy_input, outputs=dummy_output)
        dummy_model.compile(optimizer='adam', loss='binary_crossentropy')
        dummy_model.save(model_path)
        model = dummy_model # load the dummy model
        logger.info(f"Created and loaded dummy model fromừ {model_path}")


    if not Path(tokenizer_path).exists() and tokenizer is None:
        logger.warning(f"Dummy tokenizer creation as {tokenizer_path} not found. THIS IS NOT A REAL TOKENIZER.")
        # Create a dummy tokenizer
        dummy_tokenizer_obj = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
        dummy_tokenizer_obj.fit_on_texts(["this is a sample text for dummy tokenizer"])
        with open(tokenizer_path, 'wb') as f_tok:
            pickle.dump(dummy_tokenizer_obj, f_tok)
        tokenizer = dummy_tokenizer_obj # load the dummy tokenizer
        logger.info(f"Đã tạo và tải tokenizer dummy từ {tokenizer_path}")


    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- END OF FILE app.py ---