import tensorflow as tf
from tensorflow import keras
from stable_diffusion_tf.layers import PaddedConv2D
from tensorflow_addons.layers import GroupNormalization
from tensorflow.python.keras.layers import Permute, Lambda, UpSampling2D, Activation


class AttentionBlock(keras.layers.Layer):
  def __init__(self, channels):
    super().__init__()
    self.norm = GroupNormalization(epsilon=1e-5)
    self.q = PaddedConv2D(channels, 1)
    self.k = PaddedConv2D(channels, 1)
    self.v = PaddedConv2D(channels, 1)
    self.proj_out = PaddedConv2D(channels, 1)
  
  def call(self, x, **kwargs):
    h_ = self.norm(x)
    q, k, v = self.q(h_), self.k(h_), self.v(h_)
    
    # Compute attention
    b, h, w, c = q.shape
    q = tf.reshape(q, (-1, h * w, c))  # b,hw,c
    k = Permute((3, 1, 2))(k)
    k = tf.reshape(k, (-1, c, h * w))  # b,c,hw
    w_ = q @ k
    w_ = w_ * (c ** (-0.5))
    w_ = keras.activations.softmax(w_)
    
    # Attend to values
    v = Permute((3, 1, 2))(v)
    v = tf.reshape(v, (-1, c, h * w))
    w_ = Permute((2, 1))(w_)
    h_ = v @ w_
    h_ = Permute((2, 1))(h_)
    h_ = tf.reshape(h_, (-1, h, w, c))
    return x + self.proj_out(h_)


class ResnetBlock(keras.layers.Layer):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.norm1 = GroupNormalization(epsilon=1e-5)
    self.conv1 = PaddedConv2D(out_channels, 3, padding=1)
    self.norm2 = GroupNormalization(epsilon=1e-5)
    self.conv2 = PaddedConv2D(out_channels, 3, padding=1)
    self.nin_shortcut = (
      PaddedConv2D(out_channels, 1)
      if in_channels != out_channels
      else lambda x: x
    )
  
  def call(self, x, **kwargs):
    h = self.conv1(keras.activations.swish(self.norm1(x)))
    h = self.conv2(keras.activations.swish(self.norm2(h)))
    return self.nin_shortcut(x) + h


class Decoder(keras.Sequential):
  def __init__(self):
    super().__init__(
      [
        Lambda(lambda x: 1 / 0.18215 * x),    # (b, 64, 64, 4)
        PaddedConv2D(4, 1),
        PaddedConv2D(512, 3, padding=1),      # (b, 64, 64, 512)
        ResnetBlock(512, 512),
        AttentionBlock(512),
        ResnetBlock(512, 512),
        ResnetBlock(512, 512),
        ResnetBlock(512, 512),
        ResnetBlock(512, 512),
        UpSampling2D(size=(2, 2)),            # (b, 128, 128, 512)
        PaddedConv2D(512, 3, padding=1),
        ResnetBlock(512, 512),
        ResnetBlock(512, 512),
        ResnetBlock(512, 512),
        UpSampling2D(size=(2, 2)),            # (b, 256, 256, 512)
        PaddedConv2D(512, 3, padding=1),
        ResnetBlock(512, 256),                # (b, 256, 256, 256)
        ResnetBlock(256, 256),
        ResnetBlock(256, 256),
        UpSampling2D(size=(2, 2)),            # (b, 512, 512, 256)
        PaddedConv2D(256, 3, padding=1),
        ResnetBlock(256, 128),                # (b, 512, 512, 128)
        ResnetBlock(128, 128),
        ResnetBlock(128, 128),
        GroupNormalization(epsilon=1e-5),     # (b, 512, 512, 32*4)
        Activation("swish"),
        PaddedConv2D(3, 3, padding=1),        # (b, 512, 512, 3)
      ]
    )


class Encoder(keras.Sequential):
  def __init__(self):
    super().__init__(
      [
        PaddedConv2D(128, 3, padding=1),                       # (b, 512, 512, 128)
        ResnetBlock(128, 128),
        ResnetBlock(128, 128),
        PaddedConv2D(128, 3, padding=(0, 1), stride=2),         # (b, 256, 256, 128)
        ResnetBlock(128, 256),                                  # (b, 256, 256, 256)
        ResnetBlock(256, 256),
        PaddedConv2D(256, 3, padding=(0, 1), stride=2),         # (b, 128, 128, 256)
        ResnetBlock(256, 512),                                  # (b, 128, 128, 512)
        ResnetBlock(512, 512),
        PaddedConv2D(512, 3, padding=(0, 1), stride=2),         # (b, 64, 64, 512)
        ResnetBlock(512, 512),
        ResnetBlock(512, 512),
        ResnetBlock(512, 512),
        AttentionBlock(512),
        ResnetBlock(512, 512),
        GroupNormalization(epsilon=1e-5),
        Activation("swish"),
        PaddedConv2D(8, 3, padding=1),                         # (b, 64, 64, 8)
        PaddedConv2D(8, 1),
        Lambda(lambda x: x[..., :4] * 0.18215)                 # (b, 64, 64, 4)
      ]
    )
