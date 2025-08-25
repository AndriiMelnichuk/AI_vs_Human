"""
This script trains and evaluates a binary classification model using PyTorch and scikit-learn.
Workflow:
- Loads a CSV dataset containing text samples and binary labels.
- Splits the data into training and testing sets.
- Prepares custom datasets and data loaders for batching and tokenization.
- Initializes a neural network model, loss function, optimizer, and mixed precision scaler.
- Trains the model for a specified number of epochs, logging loss to TensorBoard.
- Evaluates the model on the test set after each epoch, computing the F1 score.
- Saves the model checkpoint after each epoch.
Key Components:
- CustomDataset: Handles text and label preprocessing.
- FullModel: The neural network architecture for binary classification.
- BCEWithLogitsLoss: Loss function for binary classification.
- Adam: Optimizer for model training.
- torch.amp: Enables mixed precision training for efficiency.
- SummaryWriter: Logs metrics for visualization in TensorBoard.
- f1_score: Measures model performance on the test set.
Parameters:
- abs_data_path: Path to the input CSV data file.
- model_path: Path to save the trained model.
- frac: Fraction of data used for training.
- random_state: Seed for reproducible data splits.
- batch_size: Number of samples per batch.
- epoch_count: Number of training epochs.
- max_token_count: Maximum number of tokens per input sample.
- device: Device for computation ('cuda' or 'cpu').
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch
from torch.utils.tensorboard import SummaryWriter

from ml.dataset import CustomDataset, collate_fn
from ml.model import FullModel

from sklearn.metrics import f1_score

from tqdm import tqdm


abs_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pth')
frac = 0.8
random_state = 42
batch_size = 384
epoch_count = 1
max_token_count = 512
device = 'cuda'
writer = SummaryWriter()

# --------------------------------------------------
# DATA
# --------------------------------------------------

df = pd.read_csv(abs_data_path)
train_df = df.sample(frac=frac, random_state=random_state)
test_df =  df.drop(train_df.index)

train_ds = CustomDataset(train_df['text'].reset_index(drop=True), train_df['generated'].reset_index(drop=True))
train_dl = DataLoader(
  train_ds, 
  batch_size=batch_size,
  shuffle=True, collate_fn=collate_fn
)

test_ds = CustomDataset(test_df['text'].reset_index(drop=True), test_df['generated'].reset_index(drop=True))
test_dl = DataLoader(
  test_ds, 
  batch_size=batch_size,
  shuffle=False, collate_fn=collate_fn
)

# --------------------------------------------------
# MODELING
# --------------------------------------------------

model = FullModel(2).to(device)
loss_fn = BCEWithLogitsLoss()
optimizer = Adam(model.parameters())
scaler = torch.amp.GradScaler('cuda')

for epoch in range(epoch_count):
  model.train()
  epoch_loss = 0
  train_progress = tqdm(train_dl, desc=f'[TRAIN]: Epoch: {epoch + 1}/{epoch_count}', leave=True)
  for i, (X, y) in enumerate(train_progress):
    X = X.to(device)
    X = X[:,:max_token_count]
    y = y.to(device)

    with torch.amp.autocast('cuda'):
      logits = model(X)
      loss = loss_fn(logits, y)
      
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    global_step = i + epoch * len(train_dl)
    writer.add_scalar('loss', loss.item(), global_step=global_step)
    epoch_loss += loss.item()
    train_progress.set_postfix({
      'loss': f'{loss:.4f}',
      'avg_loss': f'{epoch_loss / len(train_progress):.4f}'
    })
    

  model.eval()
  text_progress = tqdm(test_dl, desc=f'[TEST] Epoch: {epoch + 1}/{epoch_count}', leave=True)
  with torch.no_grad():
    pr_list = []
    y_list = []
    for X, y in text_progress:
      X = X.to(device)
      X = X[:,:max_token_count]
      pr_list.append(
        torch.sigmoid(model(X).to('cpu'))
      )
      y_list.append(y)
    pr = (torch.vstack(pr_list) > 0.5).to(int).numpy()
    y  = torch.vstack(y_list).to(torch.int).numpy()
    f1 = f1_score(y, pr)
  
  writer.add_scalar('f1', f1, epoch)
  print(f"TEST F1 score {epoch + 1}/{epoch_count}: {f1}")
  torch.save(model.state_dict(), model_path)
  print('-'*50)


