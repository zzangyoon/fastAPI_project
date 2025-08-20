# python만 설치
FROM python:3.11-slim

# 실제 프로그램이 동작할 폴더 생성(작업디렉토리)
WORKDIR /app

# 이미지 안으로 파일 복사(COPY)
# 환경설정
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install uvicorn fastapi
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY 3_infermain.py .

COPY model/ /app/model/
COPY imgData/ /app/imgData/

# port 설정 (EXPOSE)
EXPOSE 7070

# 띄어쓰기 단위로 CMD[]안에 하나의 문자열로 담는다
# uvicorn infermain:app --host 0.0.0.0 --port 7070
CMD ["uvicorn", "3_infermain:app", "--host", "0.0.0.0", "--port", "7070"]

# dockerhub_username : 도커허브 아이디(yoo000n)
# your_image_name : 이미지 이름
# tag : 버전정보
# docker build -t yoo000n/fastinfer:v0.1 .

# docker images

# docker run -p 임의:port번호 이미지
# docker run -p 9090:7070 yoo000n/fastinfer:v0.1
# docker가 gpu를 못쓰고 있음 (서버 os에게 gpu 쓴다고 허락 받아야함)
# 동작되고있는 container 확인
# docker ps
# docker ps -a (-a : all - 동작되었던.. 지금은 죽은)
# 아예 제거해야함 -> docker rm 54b82bff280b(CONTAINER ID) 
# docker run --rm --gpus all -p 9090:7070 yoo000n/fastinfer:v0.1
# --rm : container 죽으면 지워라 --gpus all : 모든 gpu 써라
# docker run -d --gpus all -p 9090:7070 yoo000n/fastinfer:v0.1
# -d : daemon 으로 돌림