# Sử dụng image Python chính thức
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements.txt vào container
COPY requirements.txt .

# Cài đặt các dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy các file mô hình và mã nguồn ứng dụng vào container
COPY models/ /app/models/
COPY templates/ /app/templates/
COPY app.py .

# Expose cổng 8000 cho FastAPI
EXPOSE 8000

# Command để chạy ứng dụng FastAPI với Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
