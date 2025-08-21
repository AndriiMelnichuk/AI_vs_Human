import torch
from torch import nn, Tensor, no_grad, ones, arange, sin, cos


class Encoder(nn.Module):
  def __init__(self, d_model: int = 300, num_heads: int = 5, embedding_size: int = 300):
    """
    Initializes the model with multi-head attention, layer normalization, and feed-forward layers.
    Args:
      d_model (int, optional): The dimension of the model and input to the attention mechanism. Defaults to 300.
      num_heads (int, optional): Number of attention heads in the multi-head attention layer. Defaults to 5.
      embedding_size (int, optional): Size of the input embeddings. Defaults to 300.
    Attributes:
      embedding_size (int): Stores the embedding size.
      attention (nn.MultiheadAttention): Multi-head attention layer.
      ln1 (nn.LayerNorm): First layer normalization applied to the input.
      ff1 (nn.Linear): First feed-forward linear layer expanding the dimension.
      relu (nn.ReLU): ReLU activation function.
      ff2 (nn.Linear): Second feed-forward linear layer reducing the dimension.
      ln2 (nn.LayerNorm): Second layer normalization applied after feed-forward layers.
    """

    # ln - LinearNorm
    # ff - Feed Forward (Linear)
    super().__init__()
    self.embedding_size = embedding_size

    self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    self.ln1 = nn.LayerNorm(embedding_size)
    self.ff1 = nn.Linear(d_model, d_model * 4)
    self.relu = nn.ReLU()
    self.ff2 = nn.Linear(d_model * 4, d_model)
    self.ln2 = nn.LayerNorm(embedding_size)


  def forward(self, X: Tensor):
    # Self-Attention + residual
    attention_out, _ = self.attention(X, X, X)
    out1 = self.ln1(X + attention_out)

    # FeedForward + residual
    linear_result = self.ff2(self.relu(self.ff1(out1)))
    out2 = self.ln2(out1 + linear_result)

    return out2
  

# TODO: what, if we would calculate pe in init?
class PositionalEncoding(nn.Module):
  """
  PositionalEncoding module for adding positional information to input tensors.
  This module generates sinusoidal positional encodings as described in the "Attention is All You Need" paper.
  The positional encodings are computed using sine and cosine functions of different frequencies and are
  multiplied element-wise with the input tensor.
  """

  def __init__(self):
    super().__init__()
  
  def forward(self, X: Tensor) -> Tensor:
    # batch, pos, i
    with no_grad():
      max_pos = X.shape[1]
      max_i   = X.shape[2]

      a = ones(max_pos, max_i)

      pos = arange(max_pos).unsqueeze(1) * a

      i = arange(max_i)
      i = 1 / (10_000 ** (2 * i / max_i))

      pe = pos * i
      pe = pe.to(X.device)
      pe[:,::2] = sin(pe[:,::2])
      pe[:,1::2] = cos(pe[:,1::2])

    return X * pe
  

class FullModel(nn.Module):
  def __init__(self, encoder_count: int = 5, d_model: int = 300, num_heads: int = 5):
    super().__init__()
    
    self.pe = PositionalEncoding()
    self.encoders = nn.Sequential(*[Encoder(d_model=d_model, num_heads=num_heads) for _ in range(encoder_count)])
    self.model = nn.Sequential(self.pe, self.encoders)
  
  def forward(self, X: Tensor) -> Tensor:
    return self.model.forward(X)
