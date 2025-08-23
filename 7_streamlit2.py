import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from streamlit_drawable_canvas import st_canvas

class_names = ["공명", "마동석", "카리나"]

# 모델 불러오기
# 매번 불러오면 오래걸리고 번거로움 -> 캐싱(메모리 저장) : @st.cache_resource

@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, 3)
    model.load_state_dict(torch.load("./model/mymodel.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# 이미지 전처리
def transform_image(image):
    transforms_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    return transforms_test(image).unsqueeze(0)   # (3, 224, 224)

st.title("연예인분류기 V.1")

# 업로드 이미지
upload_file = st.file_uploader("이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    image = Image.open(upload_file).convert("RGB")  # convert("RGB") : 흑백일 경우를 대비해 컬러로 변경
    st.image(image, caption="업로드 이미지", use_container_width=True)  # use_container_width=True : 이미지 자동확대

    model = load_model()        # 캐시에서 불러온 모델
    infer_img = transform_image(image)  # 전처리된 이미지

    with torch.no_grad():
        result = model(infer_img)
        preds = torch.max(result, dim=1)[1]
        pred_classname = class_names[preds.item()]
        confidence = torch.softmax(result, dim=1)[0][preds.item()].item() * 100     # 정확도(confidence)

    st.success(f"예측결과 : **{pred_classname}** ({confidence : .2f}% 확신)")


# 웹캠
cam_img = st.camera_input("웹캠")

if cam_img is not None:
    image = Image.open(cam_img).convert("RGB")  # convert("RGB") : 흑백일 경우를 대비해 컬러로 변경
    st.image(image, caption="업로드 이미지", use_container_width=True)  # use_container_width=True : 이미지 자동확대

    model = load_model()        # 캐시에서 불러온 모델
    infer_img = transform_image(image)  # 전처리된 이미지

    with torch.no_grad():
        result = model(infer_img)
        preds = torch.max(result, dim=1)[1]
        pred_classname = class_names[preds.item()]
        confidence = torch.softmax(result, dim=1)[0][preds.item()].item() * 100     # 정확도(confidence)

    st.success(f"예측결과 : **{pred_classname}** ({confidence : .2f}% 확신)")

# canvas
# uv add streamlit-drawable-canvas
canvas_img = st_canvas(
    fill_color = "white",   # 내부 색상 #001100, rgb, rgba
    stroke_width = 3,       # 펜 두께
    stroke_color = "black",
    background_color = "white",
    height = 400,
    width = 400,
    drawing_mode = "freedraw",   # 모드(freedraw, line, rect, circle, transform),
    key = "canvas"
)

if canvas_img is not None:
    image = Image.fromarray(canvas_img.image_data).convert("RGB")  # convert("RGB") : 흑백일 경우를 대비해 컬러로 변경
    st.image(image, caption="업로드 이미지", use_container_width=True)  # use_container_width=True : 이미지 자동확대

    model = load_model()        # 캐시에서 불러온 모델
    infer_img = transform_image(image)  # 전처리된 이미지

    with torch.no_grad():
        result = model(infer_img)
        preds = torch.max(result, dim=1)[1]
        pred_classname = class_names[preds.item()]
        confidence = torch.softmax(result, dim=1)[0][preds.item()].item() * 100     # 정확도(confidence)

    st.success(f"예측결과 : **{pred_classname}** ({confidence : .2f}% 확신)")