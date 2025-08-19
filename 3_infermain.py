############################################################
# inference (추론)
# 학습이 완료된 모델을 사용해 새로운 입력에 대해 예측을 수행하는 과정
############################################################
# 서버를 띄울때 모델을 불러와야함
# 대기하고 있다가 클라이언트 요청 들어오면 이미지를 모델에 넣어서 결과 출력
#   이미지 들어오면 tensor/numpy.array 로 변환후 전처리(transforms) -> 모델 학습
#   다중학습을 하기 위해 요청으로 들어온 image를 저장해야함
#   (포폴용) 사용자 아키텍처(이미지)를 따로 저장해놓고, 주마다/월마다 데이터를 업데이트 시키고, 재학습시킨다고 해야함

#########################
# uv add python-multipart
#########################

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# Server
app = FastAPI(title="ResNet34 Inference")

# Model
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=3, bias=True)

# 가중치 불러오기
model.load_state_dict(torch.load("./model/mymodel.pth"))

model.eval()    # 평가모드로 설정
model.to(device)

# transform
transforms_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# pydantic (데이터의 유효성 검증)
class response(BaseModel):
    name : str
    score : float
    type : int

@app.post("/predict", response_model=response)
async def predict(file: UploadFile=File(...)) : # file : key값 (client가 나에게 보내는 이름)
    # 이미지 열기
    image = Image.open(file.file)
    image.save("./imgData/test.jpg")            # 실무에서는 uuid, timestamp, 카운트.. 등등
    img_tensor = transforms_infer(image).unsqueeze(0).to(device)    # [3, 224, 224] -> [1, 3, 224, 224]

    # 추론
    with torch.no_grad() :
        pred = model(img_tensor)
        print(f"예측값 : {pred}")

    pred_result = torch.max(pred, dim=1)[1].item()  # 0, 1, 2
    score = nn.Softmax()(pred)[0]       # 전체를 1로 봤을때 각각의 비율 [0.03, 0.90, 0.07]
    print(f"Softmax : {score}")
    score = float(score[pred_result])
    classname = ["공명", "마동석", "카리나"]
    name = classname[pred_result]
    print(f"name : {name}")

    return response(name=name, score=score, type=pred_result)

# 처음 프로젝트 시작할때는 return 값에 dummy data 두고 진행
@app.post("/predict", response_model=response)
async def predict(file: UploadFile=File(...)) :
    return response(name="test1", score=0.123, type=1)