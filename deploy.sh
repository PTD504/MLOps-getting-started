#!/bin/bash

# Xây dựng và khởi động container
echo "Đang xây dựng và khởi động container..."
docker-compose up -d --build

# Kiểm tra container đã chạy thành công chưa
echo "Kiểm tra trạng thái container..."
docker ps | grep -i "sentiment.*api"
if [ $? -ne 0 ]; then
    echo "Lỗi: Container không chạy thành công."
    exit 1
fi
# Kiểm tra trạng thái API
echo "Kiểm tra trạng thái API..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health
if [ $? -ne 0 ]; then
    echo "Lỗi: API không phản hồi."
    exit 1
fi

echo "API đã được triển khai tại http://localhost:8000"
echo "Tài liệu API có thể truy cập tại http://localhost:8000/docs"