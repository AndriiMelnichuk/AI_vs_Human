"""
Tokenizer wrapper for Hugging Face Transformers.

This module provides a wrapper around Hugging Face's AutoTokenizer to 
tokenize single texts or batches of texts into tensors, as well as to 
access vocabulary size. It also includes utilities for handling sequences 
with proper padding.
"""

from transformers import AutoTokenizer
from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence


class Tokenizer:
  """
  Tokenizer wrapper built on top of Hugging Face's AutoTokenizer.

  Args:
    tokenizer_name (str, optional): Name of the pretrained tokenizer. 
      Defaults to "t5-small".
  """
  def __init__(self, tokenizer_name: str = 't5-small'):
    self.__tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


  def tokenize(self, text: str | list[str]) -> Tensor:
    """
    Tokenize text or a list of texts into a padded tensor of token IDs.

    If a single string is given, returns a tensor of shape [1, seq_len].
    If a list of strings is given, returns a tensor of shape [batch_size, max_seq_len],
    where sequences are padded to the same length.

    Args:
        text (str | list[str]): Input text or list of texts.

    Returns:
        Tensor: A tensor containing token IDs (batch_first).
    """
    if isinstance(text, str):
      text = '<extra_id_0>' + text
      return tensor(self.__tokenizer.encode(text)).unsqueeze(0)
    else:
      tokenized_texts = [self.tokenize(t).squeeze(0) for t in text]
      return pad_sequence(tokenized_texts, batch_first=True)

  def vocab_len(self):
    """
    Return the size of the tokenizer vocabulary.

    Returns:
      int: Vocabulary size.
    """
    return len(self.__tokenizer.vocab)

