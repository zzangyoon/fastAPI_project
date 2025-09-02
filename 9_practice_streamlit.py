import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from streamlit_drawable_canvas import st_canvas

class_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
# 모델 불러오기

# 모델 클래스
class Lenet(nn.Module) :
    def __init__(self) :
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x) :
        x = self.conv1(x)
        x = F.tanh(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.tanh(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = F.tanh(x)
        x = x.view(-1, 120)

        x = self.fc1(x)
        x = F.tanh(x)

        x = self.fc2(x)
        x = F.tanh(x)

        return x

@st.cache_resource
def load_model():
    model = Lenet()
    model.load_state_dict(torch.load("./model/mnistModel_weight.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


def transform_image(image):
    data_transform = transforms.Compose(
    [
        transforms.ToTensor(),              # tensor 변환
        transforms.Resize(32),              # 이미지 크기변경
        transforms.Normalize((0.5), (1.0))  # 정규화 (평균, 표준편차)
    ]
)
    return data_transform(image).unsqueeze(0)   # (3, 224, 224)

st.title("숫자 분류기 V.1")

# 업로드 이미지
upload_file = st.file_uploader("이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    image = Image.open(upload_file).convert("L")
    st.image(image, caption="업로드 이미지", use_container_width=True)  # use_container_width=True : 이미지 자동확대

    model = load_model()        # 캐시에서 불러온 모델
    infer_img = transform_image(image)  # 전처리된 이미지

    with torch.no_grad():
        result = model(infer_img)
        preds = result.argmax(dim=1).item()
        print("preds ::: ", preds)
    #     preds = torch.max(result, dim=1)[1]
    #     pred_classname = class_names[preds.item()]
    #     confidence = torch.softmax(result, dim=1)[0][preds.item()].item() * 100     # 정확도(confidence)

    st.success(f"예측결과 : **{preds}**")
# ({confidence : .2f}% 확신)


# canvas
# uv add streamlit-drawable-canvas
canvas_img = st_canvas(
    fill_color = "white",   # 내부 색상 #001100, rgb, rgba
    stroke_width = 3,       # 펜 두께
    stroke_color = "white",
    background_color = "black",
    height = 400,
    width = 400,
    drawing_mode = "freedraw",   # 모드(freedraw, line, rect, circle, transform),
    key = "canvas"
)

if canvas_img is not None:
    image = Image.fromarray(canvas_img.image_data).convert("L")
    st.image(image, caption="업로드 이미지", use_container_width=True)  # use_container_width=True : 이미지 자동확대

    model = load_model()        # 캐시에서 불러온 모델
    infer_img = transform_image(image)  # 전처리된 이미지

    with torch.no_grad():
        result = model(infer_img)
        # preds = result.argmax(dim=1)
        preds = torch.max(result, dim=1)
        # pred_classname = class_names[preds]
        print("preds ::: ", preds)
    #     preds = torch.max(result, dim=1)[1]
    #     pred_classname = class_names[preds.item()]
    #     confidence = torch.softmax(result, dim=1)[0][preds.item()].item() * 100     # 정확도(confidence)

    # st.success(f"예측결과 : **{pred_classname}**")