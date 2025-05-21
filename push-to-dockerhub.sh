#!/bin/bash

# Lấy username từ tham số dòng lệnh hoặc yêu cầu người dùng nhập
if [ -z "$1" ]; then
    echo "Vui lòng nhập tên người dùng Docker Hub của bạn:"
    read USERNAME
else
    USERNAME=$1
fi
# Kiểm tra nếu người dùng chưa nhập username
if [ "$USERNAME" = "your-dockerhub-username" ] || [ -z "$USERNAME" ]; then
    echo "Vui lòng cung cấp tên người dùng Docker Hub thực tế."
    echo "Sử dụng: ./push-to-dockerhub.sh [username]"
    exit 1
fi
# Đăng nhập vào Docker Hub (bạn sẽ được yêu cầu nhập mật khẩu)
echo "Đăng nhập vào Docker Hub..."
docker login

# Gắn tag cho image local
echo "Gắn tag cho image..."
docker tag sentiment-analysis-api:latest $USERNAME/sentiment-analysis-api:latest

# Đẩy image lên Docker Hub
echo "Đẩy image lên Docker Hub..."
docker push $USERNAME/sentiment-analysis-api:latest

echo "Hoàn tất! Image đã được đẩy lên Docker Hub như $USERNAME/sentiment-analysis-api:latest"