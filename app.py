import os
import pickle
import time
import psutil
import numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
import joblib
import re

# Import monitoring libraries
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_fastapi_instrumentator import Instrumentator
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Monitoring libraries not available. Install with: pip install prometheus-client prometheus-fastapi-instrumentator")

# ================================================================================
# CONFIGURATION
# ================================================================================
MODEL_DIR = 'models'
MAX_LEN = 200
APP_VERSION = "2.0.0"
APP_NAME = "Sentiment Analysis API"

# ================================================================================
# PYDANTIC MODELS
# ================================================================================
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    model_type: str = Field(default="traditional", description="Model type: traditional or deep")

# ================================================================================
# PROMETHEUS METRICS (if available)
# ================================================================================
if MONITORING_AVAILABLE:
    PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions made', ['model_type', 'sentiment'])
    PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency', ['model_type'])
    CONFIDENCE_SCORE = Histogram('ml_prediction_confidence', 'Prediction confidence', ['model_type'])
    CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
    MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Memory usage percentage')

# ================================================================================
# FASTAPI APP SETUP
# ================================================================================
app = FastAPI(
    title=APP_NAME,
    description="API phân tích cảm xúc văn bản sử dụng ML và Deep Learning với monitoring",
    version=APP_VERSION
)

# Setup monitoring
if MONITORING_AVAILABLE:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================================================================================
# MODEL LOADING
# ================================================================================
# Load traditional model
try:
    traditional_model_path = os.path.join(MODEL_DIR, 'logistic_regression.joblib')
    traditional_model = joblib.load(traditional_model_path)
    print(f"✅ Loaded traditional model from {traditional_model_path}")
except Exception as e:
    print(f"❌ Cannot load traditional model: {e}")
    traditional_model = None

# Load LSTM model and tokenizer
try:
    lstm_model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✅ Loaded LSTM model from {lstm_model_path}")
except Exception as e:
    print(f"❌ Cannot load LSTM model: {e}")
    lstm_model = None
    tokenizer = None

# ================================================================================
# HTML TEMPLATE CREATION
# ================================================================================
index_html_path = templates_dir / "index.html"
if not index_html_path.exists():
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        textarea {
            width: 100%;
            height: 120px;
            margin: 15px 0;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
            transition: border-color 0.3s;
        }
        textarea:focus {
            border-color: #4CAF50;
            outline: none;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        }
        .button-container {
            text-align: center;
            margin: 25px 0;
        }
        button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }
        button:disabled {
            background: #cccccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .result {
            margin-top: 25px;
            padding: 25px;
            border-radius: 12px;
            display: none;
            animation: slideIn 0.5s ease-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .positive {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border-left: 5px solid #28a745;
        }
        .negative {
            background: linear-gradient(135deg, #f8d7da, #f1b0b7);
            border-left: 5px solid #dc3545;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .highlight {
            background: linear-gradient(45deg, #fff176, #ffeb3b);
            padding: 3px 6px;
            border-radius: 4px;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(255,235,59,0.3);
        }
        .loading {
            display: none;
            text-align: center;
            margin: 25px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Sentiment Analysis</h1>
        <p class="subtitle">Analyze text sentiment using Machine Learning & Deep Learning</p>
        
        <textarea id="text-input" placeholder="Enter your text here... (e.g., 'This movie is absolutely amazing!')"></textarea>
        
        <div class="button-container">
            <button onclick="predict('traditional')" id="btn-traditional">📊 Traditional ML</button>
            <button onclick="predict('deep')" id="btn-deep">🧠 Deep Learning</button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing sentiment...</p>
        </div>
        
        <div id="result" class="result">
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="sentiment">-</div>
                    <div class="metric-label">Sentiment</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="confidence">-</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="model-used">-</div>
                    <div class="metric-label">Model Used</div>
                </div>
            </div>
            
            <div>
                <strong>📝 Text with highlighted keywords:</strong>
                <div id="highlighted-text" style="margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.5); border-radius: 8px; line-height: 1.8;"></div>
            </div>
        </div>
        
        <div class="footer">
            <p>MLOps Sentiment Analysis API v2.0 | Powered by FastAPI & TensorFlow</p>
        </div>
    </div>

    <script>
        function predict(modelType) {
            const text = document.getElementById('text-input').value.trim();
            if (!text) {
                alert('Please enter some text');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('btn-traditional').disabled = true;
            document.getElementById('btn-deep').disabled = true;
            
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
            .then(response => {
                if (!response.ok) {
                    throw new Error('API Error: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Update results
                document.getElementById('sentiment').textContent = data.sentiment;
                document.getElementById('confidence').textContent = data.probability + '%';
                document.getElementById('model-used').textContent = modelType.charAt(0).toUpperCase() + modelType.slice(1);
                
                // Display result
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'result ' + (data.sentiment.toLowerCase() === 'positive' ? 'positive' : 'negative');
                
                // Highlight keywords
                let textWithHighlights = text;
                if (data.highlighted_words && data.highlighted_words.length > 0) {
                    data.highlighted_words.forEach(word => {
                        const regex = new RegExp('\\\\b' + word + '\\\\b', 'gi');
                        textWithHighlights = textWithHighlights.replace(regex, `<span class="highlight">${word}</span>`);
                    });
                }
                
                document.getElementById('highlighted-text').innerHTML = textWithHighlights;
                
                // Re-enable buttons
                document.getElementById('btn-traditional').disabled = false;
                document.getElementById('btn-deep').disabled = false;
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('btn-traditional').disabled = false;
                document.getElementById('btn-deep').disabled = false;
                alert('Error: ' + error.message);
            });
        }
        
        // Enter key support
        document.getElementById('text-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                predict('traditional');
            }
        });
    </script>
</body>
</html>
        ''')

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================
def clean_text(text):
    """Clean text for traditional model"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def predict_traditional(text):
    """Predict using traditional model"""
    if traditional_model is None:
        raise HTTPException(status_code=503, detail="Traditional model not available")
        
    cleaned_text = clean_text(text)
    prediction = traditional_model.predict([cleaned_text])[0]
    
    if hasattr(traditional_model, 'predict_proba'):
        probability = traditional_model.predict_proba([cleaned_text])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        prob = probability[1] if prediction == 1 else probability[0]
    else:
        sentiment = "Positive" if prediction == 1 else "Negative"
        prob = 0.8  # Default confidence
    
    return sentiment, float(prob)

def predict_lstm(text):
    """Predict using LSTM model"""
    if lstm_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="LSTM model not available")
        
    sequences = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    
    prediction = lstm_model.predict(padded_seq, verbose=0)[0][0]
    
    if prediction > 0.5:
        sentiment = "Positive"
        prob = float(prediction)
    else:
        sentiment = "Negative"
        prob = 1 - float(prediction)
    
    return sentiment, prob

def get_highlighted_words(text, sentiment):
    """Get words to highlight"""
    words = text.lower().split()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'like', 'fantastic', 'awesome']
    negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'disappointed', 'poor', 'horrible', 'disgusting']
    
    if sentiment == "Positive":
        return [word for word in words if word in positive_words]
    else:
        return [word for word in words if word in negative_words]

def update_metrics():
    """Update system metrics"""
    if MONITORING_AVAILABLE:
        try:
            CPU_USAGE.set(psutil.cpu_percent())
            MEMORY_USAGE.set(psutil.virtual_memory().percent)
        except:
            pass

# ================================================================================
# API ENDPOINTS
# ================================================================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_data: TextInput):
    """Predict sentiment"""
    start_time = time.time()
    
    try:
        # Validate model type
        if input_data.model_type not in ['traditional', 'deep']:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'traditional' or 'deep'")
        
        # Make prediction
        if input_data.model_type == 'traditional':
            sentiment, probability = predict_traditional(input_data.text)
        else:  # deep
            sentiment, probability = predict_lstm(input_data.text)
        
        # Get highlighted words
        highlighted_words = get_highlighted_words(input_data.text, sentiment)
        
        # Update metrics
        prediction_time = time.time() - start_time
        if MONITORING_AVAILABLE:
            PREDICTION_COUNTER.labels(model_type=input_data.model_type, sentiment=sentiment.lower()).inc()
            PREDICTION_LATENCY.labels(model_type=input_data.model_type).observe(prediction_time)
            CONFIDENCE_SCORE.labels(model_type=input_data.model_type).observe(probability)
        
        return {
            'sentiment': sentiment,
            'probability': round(probability * 100, 2),
            'highlighted_words': highlighted_words,
            'model_type': input_data.model_type,
            'prediction_time': round(prediction_time, 4)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    update_metrics()
    
    models_status = {
        "traditional": traditional_model is not None,
        "lstm": lstm_model is not None and tokenizer is not None
    }
    
    system_info = {}
    try:
        system_info = {
            "cpu_usage": round(psutil.cpu_percent(), 2),
            "memory_usage": round(psutil.virtual_memory().percent, 2)
        }
    except:
        system_info = {"cpu_usage": 0, "memory_usage": 0}
    
    status = "healthy" if any(models_status.values()) else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "models": models_status,
        "system": system_info,
        "version": APP_VERSION,
        "monitoring": MONITORING_AVAILABLE
    }

@app.get("/info")
async def get_info():
    """API information"""
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "description": "Sentiment Analysis API with ML and Deep Learning",
        "models_available": {
            "traditional": traditional_model is not None,
            "lstm": lstm_model is not None and tokenizer is not None
        },
        "features": [
            "Traditional ML (Logistic Regression)",
            "Deep Learning (LSTM)", 
            "Real-time predictions",
            "Interactive web interface",
            "Health monitoring",
            "Prometheus metrics" if MONITORING_AVAILABLE else "Basic monitoring"
        ],
        "endpoints": {
            "home": "/",
            "predict": "/predict",
            "health": "/health",
            "info": "/info",
            "metrics": "/metrics" if MONITORING_AVAILABLE else "Not available",
            "docs": "/docs"
        }
    }

if MONITORING_AVAILABLE:
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics"""
        update_metrics()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ================================================================================
# STARTUP EVENTS
# ================================================================================
@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print(f"\n🚀 Starting {APP_NAME} v{APP_VERSION}")
    print(f"📊 Traditional model: {'✅' if traditional_model else '❌'}")
    print(f"🧠 LSTM model: {'✅' if (lstm_model and tokenizer) else '❌'}")
    print(f"📈 Monitoring: {'✅' if MONITORING_AVAILABLE else '❌'}")
    print(f"🌐 Access at: http://localhost:8000")
    print(f"📚 API docs: http://localhost:8000/docs\n")

# ================================================================================
# MAIN
# ================================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")