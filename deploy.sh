#!/bin/bash

# Thông tin kết nối đến server
SERVER_USER=${1:-user}  # Tham số 1: tên người dùng SSH
SERVER_IP=${2:-localhost}  # Tham số 2: địa chỉ IP server
SERVER_PATH=${3:-/home/$SERVER_USER/sentiment-api}  # Tham số 3: đường dẫn triển khai
DOCKER_HUB_USERNAME=${4:-22521571}  # Tham số 4: username Docker Hub

# Hiển thị hướng dẫn nếu không có đủ tham số
if [ "$SERVER_IP" = "localhost" ]; then
    echo "Sử dụng: ./deploy.sh [username] [server-ip] [server-path] [dockerhub-username]"
    echo "Ví dụ: ./deploy.sh ubuntu 192.168.1.100 /home/ubuntu/sentiment-api 22521571"
    exit 1
fi

echo "=== TRIỂN KHAI LÊN SERVER ==="
echo "Server: $SERVER_USER@$SERVER_IP"
echo "Đường dẫn: $SERVER_PATH"
echo "Docker Hub Username: $DOCKER_HUB_USERNAME"

# Tạo file docker-compose-hub.yml
echo "Tạo file docker-compose-hub.yml..."
cat > docker-compose-hub.yml << EOF
services:
  sentiment-api:
    image: $DOCKER_HUB_USERNAME/sentiment-analysis-api:latest
    ports:
      - "8000:8000"
    restart: unless-stopped
EOF

# Tạo thư mục trên server
echo "Tạo thư mục trên server..."
ssh $SERVER_USER@$SERVER_IP "mkdir -p $SERVER_PATH"

# Sao chép file docker-compose-hub.yml lên server
echo "Sao chép file cấu hình lên server..."
scp docker-compose-hub.yml $SERVER_USER@$SERVER_IP:$SERVER_PATH/docker-compose.yml

# Triển khai container trên server
echo "Triển khai container trên server..."
ssh $SERVER_USER@$SERVER_IP "cd $SERVER_PATH && docker-compose pull && docker-compose up -d"

# Kiểm tra container đã chạy chưa
echo "Kiểm tra trạng thái container trên server..."
ssh $SERVER_USER@$SERVER_IP "docker ps | grep -i sentiment-api"
if [ $? -ne 0 ]; then
    echo "Lỗi: Container không chạy thành công trên server."
    exit 1
fi

# Chờ API khởi động
echo "Chờ API khởi động (khoảng 10 giây)..."
sleep 10

# Kiểm tra API đã hoạt động chưa
echo "Kiểm tra trạng thái API..."
API_STATUS=$(ssh $SERVER_USER@$SERVER_IP "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/info || echo 'error'")

if [ "$API_STATUS" = "200" ]; then
    echo "API đã được triển khai thành công!"
    echo "Web interface: http://$SERVER_IP:8000"
    echo "API documentation: http://$SERVER_IP:8000/docs"
else
    echo "Lỗi: API không phản hồi đúng cách (HTTP status: $API_STATUS)."
    echo "Kiểm tra logs container:"
    ssh $SERVER_USER@$SERVER_IP "docker logs \$(docker ps -q --filter 'name=sentiment-api')"
    exit 1
fi

# Hiển thị cách test API
echo "
=== TEST API TỪ MÁY LOCAL ===
# Kiểm tra thông tin API
curl http://$SERVER_IP:8000/info

# Phân tích cảm xúc với mô hình truyền thống
curl -X POST http://$SERVER_IP:8000/predict \\
  -H \"Content-Type: application/json\" \\
  -d '{\"text\":\"This movie was great!\", \"model_type\":\"traditional\"}'
"
