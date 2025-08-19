#########################
# uv add fastapi uvicorn
#########################

from fastapi import FastAPI

app = FastAPI()

@app.get("/")       # http://127.0.0.1/
def read_root() :
    return {"result" : "", "score":"0.97"}

################################
# 서버 실행
# uvicorn fastmain:app --reload
################################
################################
# option : --port 8003
################################

@app.get("/image")
def make_image():
    return {"result" : "생성완료"}

@app.get("/chatbot")
def chatbot():
    return {"result" : "안녕하세요 저는 ChatGPT 입니다."}

@app.get("/video")
def make_video():
    return {"result" : "동영상 생성이 완료되었습니다."}