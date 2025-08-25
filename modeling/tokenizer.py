"""
Tokenizer wrapper for Hugging Face Transformers.

This module provides a wrapper around Hugging Face's AutoTokenizer to 
tokenize single texts or batches of texts into tensors, as well as to 
access vocabulary size. It also includes utilities for handling sequences 
with proper padding.
"""

from transformers import AutoTokenizer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class Tokenizer:
  """
  Tokenizer wrapper built on top of Hugging Face's AutoTokenizer.

  Args:
    tokenizer_name (str, optional): Name of the pretrained tokenizer. 
      Defaults to "t5-small".
  """
  def __init__(self, tokenizer_name: str = 't5-small'):
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


  def tokenize_text(self, text: str) -> Tensor:
    """
    Tokenize a single text string into a tensor of token IDs.

    Args:
      text (str): Input text.

    Returns:
      Tensor: Tensor containing token IDs.
    """
    return self.tokenizer.encode(text)


  def tokenize_texts(self, texts: list[str]) -> Tensor:
    """
    Tokenize a list of text strings and pad them to equal length.

    Args:
      texts (List[str]): List of input texts.

    Returns:
      Tensor: Padded tensor of token IDs with shape (batch_size, seq_len).
    """
    tokenized_texts = [self.tokenize_text(text) for text in texts]
    return pad_sequence(tokenized_texts, batch_first=True)


  def vocab_len(self):
    """
    Return the size of the tokenizer vocabulary.

    Returns:
      int: Vocabulary size.
    """
    return len(self.tokenizer.vocab)

