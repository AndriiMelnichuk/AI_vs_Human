from ml.tokenizer import Tokenizer
from ml.model import FullModel

import os
import torch

class ModelWrapper:
  def __init__(
    self, model_path: str = 'model.pth', device: str = None, 
    tokenizer: Tokenizer = Tokenizer(), max_token_count: int = 512
  ):
    script_path = os.path.dirname(__file__)
    model_path = os.path.join(script_path, '..', 'models', model_path)

    if device is None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = 'cpu'

    model = FullModel(2)
    model.load_state_dict(torch.load(model_path, map_location=self.device))
    self.model = model.to(self.device)
    self.model.eval()

    self.tokenizer = tokenizer
    self.max_token_count = max_token_count

  
  def predict(self, text: list[str] | str) -> list[float]:
    tokens = self.tokenizer.tokenize(text)
    X = tokens[:,:self.max_token_count].to(self.device)

    res = self.model(X)
    logits = torch.sigmoid(res)
    
    return logits.reshape(-1).tolist()