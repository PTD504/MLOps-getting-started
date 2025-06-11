# Sử dụng Python 3.9 vì nó tương thích tốt với TensorFlow và các thư viện ML khác
FROM python:3.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói phụ thuộc cần thiết để biên dịch một số gói Python
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các thư viện Python cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Copy các thư mục cần thiết cho ứng dụng
COPY app.py .
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY models/ ./models/

# Tạo thư mục để lưu trữ mô hình nếu chưa có
RUN mkdir -p models
# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Khởi chạy ứng dụng FastAPI bằng Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
