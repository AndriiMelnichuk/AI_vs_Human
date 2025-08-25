"""
Transformer-based encoder model with positional encoding.

This module defines:
- Encoder: A single Transformer encoder block with self-attention and feedforward layers.
- PositionalEncoding: A sinusoidal positional encoding module.
- FullModel: A complete model combining embedding, positional encoding, multiple encoders, 
  and a feedforward classifier head.
"""

from torch import nn, Tensor, no_grad, ones, arange, sin, cos


class Encoder(nn.Module):
  """
  Transformer encoder block with self-attention and feedforward layers.

  Args:
    d_model (int): Dimensionality of the model embeddings.
    num_heads (int): Number of attention heads.
  """
  def __init__(self, d_model: int = 300, num_heads: int = 5):
    super().__init__()
    self.ln1 = nn.LayerNorm(d_model)
    self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    self.ff1 = nn.Linear(d_model, d_model * 4)
    self.gelu = nn.GELU()
    self.ff2 = nn.Linear(d_model * 4, d_model)
    self.ln2 = nn.LayerNorm(d_model)


  def forward(self, X: Tensor) -> Tensor:
    """
    Forward pass of the encoder block.

    Args:
      x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

    Returns:
      Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    # Self-Attention + residual
    ln_X = self.ln1(X)
    attention_out, _ = self.attention(ln_X, ln_X, ln_X)
    out1 = X + attention_out

    # FeedForward + residual
    out1_ln = self.ln2(out1)
    linear_result = self.ff2(self.gelu(self.ff1(out1_ln)))
    out2 = out1 + linear_result

    return out2
  

# TODO: what, if we would calculate pe in init?
class PositionalEncoding(nn.Module):
  """
  Sinusoidal positional encoding.

  Adds positional information to the embeddings using sine and cosine 
  functions, as described in "Attention Is All You Need".
  """

  def __init__(self):
    super().__init__()
  
  def forward(self, X: Tensor) -> Tensor:
    """
    Forward pass for positional encoding.

    Args:
      x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

    Returns:
      Tensor: Positionally encoded tensor of the same shape.
    """
    
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
  """
  Full Transformer-like model with embedding, positional encoding,
  stacked encoders, and a feedforward classification head.

  Args:
    encoder_count (int): Number of encoder blocks.
    d_model (int): Dimensionality of embeddings.
    num_heads (int): Number of attention heads.
    num_embeddings (int): Number of embeddings.
  """
  def __init__(self, encoder_count: int = 5, d_model: int = 300, num_heads: int = 5, num_embeddings: int = 32_100):
    super().__init__()
    
    self.embedding = nn.Embedding(num_embeddings, 300)
    self.pe = PositionalEncoding()
    self.encoders = nn.Sequential(
      *[Encoder(d_model=d_model, num_heads=num_heads) for _ in range(encoder_count)]
    )
    self.ln1 = nn.Linear(300,200)
    self.relu = nn.ReLU()
    self.ln2 = nn.Linear(200, 1)
  
  def forward(self, X: Tensor) -> Tensor:
    """
    Forward pass of the full model.

    Args:
      x (Tensor): Input tensor of token IDs (batch_size, seq_len).

    Returns:
      Tensor: Logits tensor of shape (batch_size, 1).
    """
    emb = self.embedding(X)
    pe = self.pe(emb)
    enc_res = self.encoders(pe)

    logits = self.relu(self.ln1(enc_res[:,0,:])) # CLS pooling
    logits = self.ln2(logits)
    return logits