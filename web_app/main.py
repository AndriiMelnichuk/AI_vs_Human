from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from ml.interface import ModelWrapper
import os

app = FastAPI()
model = ModelWrapper()

file_path = os.path.dirname(__file__)
static_path = os.path.join(file_path, 'static')
templates_path = os.path.join(file_path, 'templates')

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

class PredictRequest(BaseModel):
  text: str | list[str]
  
  @field_validator('text')
  def check_not_empty(cls, v):
    if isinstance(v, str):
      if not v.strip():
        raise ValueError('Text cannot be empty')
    elif isinstance(v, list):
      if not v or any(not isinstance(i, str) or not i.strip() for i in v):
        raise ValueError('List cannot be empty and must contain non-empty strings')
    else:
      raise ValueError('Invalid input type')
    return v


@app.post('/predict')
def predict(request: PredictRequest):
  try:
    text = request.text
    prediction = model.predict(text)
    return {'input': text, 'prediction': prediction}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



@app.get('/health')
def health():
  return {'status': 'ok'}


@app.get('/model-info')
def model_info():
  return {
    'version': '1.0',
    'trained_on': '2025-08-25',
    'f1_score': 0.98,
  }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})