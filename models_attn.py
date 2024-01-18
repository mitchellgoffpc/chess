import jax.numpy as jnp
import flax.linen as nn

class Attention(nn.Module):
  num_heads: int
  embed_size: int

  def setup(self):
    self.norm = nn.LayerNorm()
    self.q = nn.Dense(self.embed_size)
    self.k = nn.Dense(self.embed_size)
    self.v = nn.Dense(self.embed_size)
    self.out = nn.Dense(self.embed_size)

  def __call__(self, x, train=False):
    B, T, C = x.shape
    H = self.num_heads
    HC = C // self.num_heads

    x = self.norm(x)
    q = self.q(x).reshape(B, T, H, HC)
    k = self.k(x).reshape(B, T, H, HC)
    v = self.v(x).reshape(B, T, H, HC)
    x = nn.dot_product_attention(q, k, v).reshape(B, T, C)
    return self.out(x)

class MLP(nn.Module):
  embed_size: int

  def setup(self):
    self.norm = nn.LayerNorm()
    self.fc1 = nn.Dense(self.embed_size * 2)
    self.fc2 = nn.Dense(self.embed_size)

  def __call__(self, x, train=False):
    x = self.norm(x)
    x = nn.gelu(self.fc1(x))
    x = self.fc2(x)
    return x

class Block(nn.Module):
  num_heads: int
  embed_size: int

  def setup(self):
    self.attn = Attention(self.num_heads, self.embed_size)
    self.mlp = MLP(self.embed_size)

  def __call__(self, x, train=False):
    x = x + self.attn(x, train=train)
    x = x + self.mlp(x, train=train)
    return x

class ChessModel(nn.Module):
  num_layers: int
  num_heads: int
  embed_size: int

  def setup(self):
    self.embed_tokens = nn.Embed(13, self.embed_size)
    self.embed_pos = nn.Embed(64, self.embed_size)
    self.blocks = [Block(self.num_heads, self.embed_size) for _ in range(self.num_layers)]
    self.norm = nn.LayerNorm()
    self.value = nn.Dense(1)
    self.policy = nn.Dense(64 * 144)
    self.pos_tokens = jnp.arange(64)[None]

  def __call__(self, x, train=False):
    embed_t = self.embed_tokens(x.reshape(-1, 64))
    embed_p = self.embed_pos(self.pos_tokens)
    x = embed_t + embed_p
    for block in self.blocks:
      x = block(x, train=train)
    x = self.norm(x.mean(axis=1))
    v = self.value(x)
    p = self.policy(x)
    return v, p
