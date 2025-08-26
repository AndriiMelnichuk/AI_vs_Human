from fastapi import FastAPI
from pydantic import BaseModel
from ml.interface import ModelWrapper

app = FastAPI()
model = ModelWrapper()

class PredictRequest(BaseModel):
  text: str | list[str]


@app.post('/predict')
def predict(request: PredictRequest):
  text = request.text
  prediction = model.predict(text)
  return {'input': text, 'prediction': prediction}


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