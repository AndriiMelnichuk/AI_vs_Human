"""
Custom Dataset and collate function for PyTorch models.

This module provides a dataset class for handling text data and corresponding 
targets, using a custom Tokenizer for preprocessing. It also implements a 
collate function for preparing mini-batches with padded sequences.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from ml.tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
  """
  Custom dataset for text and target processing.

  Args:
    text (pd.Series): Series containing text data.
    target (pd.Series): Series containing target labels.
    tokenizer (Tokenizer, optional): Tokenizer instance for text preprocessing. Defaults to Tokenizer().
  """

  def __init__(self, text: pd.Series, target: pd.Series, tokenizer: Tokenizer = Tokenizer()):
    self.text = text
    self.target = target
    self.tokenizer = tokenizer


  def __len__(self):
    """Return the total number of samples."""
    return len(self.target)
  
  
  def __getitem__(self, index):
    """
    Retrieve tokenized text and the corresponding target.

    Args:
      index (int): Index of the sample.

    Returns:
      tuple[Tensor, Tensor]: A tuple containing:
        - Tensor with tokenized text.
        - Tensor with the corresponding target.
    """
    tokens = self.tokenizer.tokenize_text('<extra_id_0> ' + self.text.iloc[index])
    X = torch.tensor(tokens)
    y = torch.tensor(self.target.iloc[index])

    return X, y
  

def collate_fn(batch):
  """
  Collate function for creating mini-batches with padding.

  Args:
    batch (list[tuple[Tensor, Tensor]]): A list of (X, y) tuples.

  Returns:
    tuple[Tensor, Tensor]: A tuple containing:
      - Padded tensor of input sequences (X).
      - Tensor of stacked targets (y).
  """
  X = pad_sequence([item[0] for item in batch], batch_first=True)
  y = torch.stack([item[1] for item in batch]).unsqueeze(1).to(torch.float)
  return X, y