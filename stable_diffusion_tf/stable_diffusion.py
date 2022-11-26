import numpy as np
from tqdm import tqdm
import math
from typing import List
from functools import lru_cache
import tensorflow as tf
from tensorflow import keras

from stable_diffusion_tf.autoencoder_kl import Decoder, Encoder
from stable_diffusion_tf.diffusion_model import UNetModel
from stable_diffusion_tf.clip_encoder import CLIPTextTransformer
from stable_diffusion_tf.clip_tokenizer import SimpleTokenizer
from stable_diffusion_tf.constants import _UNCONDITIONAL_TOKENS, _ALPHAS_CUMPROD
from PIL import Image
import base64
from io import BytesIO

MAX_TEXT_LEN = 77


class StableDiffusion:
  def __init__(self, img_height=1000, img_width=1000,
               jit_compile=False,
               download_weights=True):
    self.img_height = img_height
    self.img_width = img_width
    self.tokenizer = SimpleTokenizer()
    
    text_encoder, diffusion_model, decoder, encoder = get_models(img_height, img_width,
                                                                 download_weights=download_weights)
    self.text_encoder = text_encoder
    self.diffusion_model = diffusion_model
    self.decoder = decoder
    self.encoder = encoder
    
    if jit_compile:
      self.text_encoder.compile(jit_compile=True)
      self.diffusion_model.compile(jit_compile=True)
      self.decoder.compile(jit_compile=True)
      self.encoder.compile(jit_compile=True)
    
    self.dtype = tf.float32
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
      self.dtype = tf.float16
  
  def generate(
      self,
      prompt,
      negative_prompt=None,
      batch_size=1,
      num_steps=25,
      unconditional_guidance_scale=7.5,
      temperature=1,
      seed=None,
      input_image=None,
      input_mask=None,
      input_image_strength=0.5,
      progress_call_back=None
  ):
    # 1) 将条件输入文本(prompt) Tokenize, prompt不能太长, 要小于77个token
    inputs: List[int] = self.tokenizer.encode(prompt)  # a list of int, start with 49406, end with 49407
    assert len(inputs) < MAX_TEXT_LEN, "Prompt is too long (should be < 77 tokens)"
    phrase = inputs + [49407] * (MAX_TEXT_LEN - len(inputs))  # padding to MAX_TEXT_LEN
    phrase = np.array(phrase)[None].astype("int32")  # shape=(1, MAX_TEXT_LEN)
    phrase = np.repeat(phrase, batch_size, axis=0)
    
    # Encode prompt tokens (and their positions) into a "context vector"
    pos_ids = np.array(list(range(MAX_TEXT_LEN)))[None].astype("int32")  # shape=(1, MAX_TEXT_LEN)
    pos_ids = np.repeat(pos_ids, batch_size, axis=0)
    context = self.text_encoder.predict_on_batch([phrase, pos_ids])
    
    input_image_tensor, input_image_array = None, None
    if input_image is not None:
      if type(input_image) is str:
        if tf.io.gfile.exists(input_image):
          input_image = Image.open(input_image)
        elif input_image.startswith('data:image/'):
          b64_content = input_image.split(',', maxsplit=1)[-1]
          input_image = Image.open(BytesIO(base64.b64decode(b64_content)))
        else:
          raise Exception("input_image error")
        input_image = input_image.resize((self.img_width, self.img_height))
      
      elif type(input_image) is np.ndarray:
        input_image = np.resize(input_image, (self.img_height, self.img_width, input_image.shape[2]))
      
      input_image_array = np.array(input_image, dtype=np.float32)[None, ..., :3]
      input_image_tensor = tf.cast((input_image_array / 255.0) * 2 - 1, self.dtype)
    
    latent_mask, input_mask_array, latent_mask_tensor = None, None, None
    if type(input_mask) is str:
      if tf.io.gfile.exists(input_mask):
        input_mask = Image.open(input_mask)
      elif input_mask.startswith('data:image/'):
        b64_content = input_mask.split(',', maxsplit=1)[-1]
        input_mask = Image.open(BytesIO(base64.b64decode(b64_content)))
      else:
        raise Exception("input_mask error")
      input_mask = input_mask.resize((self.img_width, self.img_height))
      input_mask_array = np.array(input_mask, dtype=np.float32)[None, ..., None]
      input_mask_array = input_mask_array / 255.0
      
      latent_mask = input_mask.resize((self.img_width // 8, self.img_height // 8))
      latent_mask = np.array(latent_mask, dtype=np.float32)[None, ..., None]
      latent_mask = 1 - (latent_mask.astype("float") / 255.0)
      latent_mask_tensor = tf.cast(tf.repeat(latent_mask, batch_size, axis=0), self.dtype)
    
    # Tokenize negative prompt or use default padding tokens
    unconditional_tokens = _UNCONDITIONAL_TOKENS
    if negative_prompt is not None:
      inputs = self.tokenizer.encode(negative_prompt)
      assert len(inputs) < MAX_TEXT_LEN, "Negative prompt is too long (should be < 77 tokens)"
      unconditional_tokens = inputs + [49407] * (MAX_TEXT_LEN - len(inputs))
    
    # Encode unconditional tokens (and their positions into an
    # "unconditional context vector"
    unconditional_tokens = np.array(unconditional_tokens)[None].astype("int32")
    unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
    unconditional_context = self.text_encoder.predict_on_batch(
      [unconditional_tokens, pos_ids]
    )
    timesteps = np.arange(1, 1000, 1000 // num_steps)
    input_img_noise_t: int = timesteps[int(len(timesteps) * input_image_strength)]
    latent, alphas, alphas_prev = self.get_starting_parameters(
      timesteps, batch_size, seed, input_image=input_image_tensor, input_img_noise_t=input_img_noise_t
    )
    
    if input_image is not None:
      timesteps = timesteps[: int(len(timesteps) * input_image_strength)]
    
    # Diffusion stage
    progbar = tqdm(list(enumerate(timesteps))[::-1])  # 从后向前
    for index, timestep in progbar:
      progbar.set_description(f"{index:3d} {timestep:3d}")
      e_t = self.get_model_output(
        latent,
        timestep,
        context,
        unconditional_context,
        unconditional_guidance_scale,
        batch_size,
      )
      a_t, a_prev = alphas[index], alphas_prev[index]
      latent, pred_x0 = self.get_x_prev_and_pred_x0(
        latent, e_t, index, a_t, a_prev, temperature, seed
      )
      
      if input_mask is not None and input_image is not None:
        # If mask is provided, noise at current timestep will be added to input image.
        # The intermediate latent will be merged with input latent.
        latent_orgin, alphas, alphas_prev = self.get_starting_parameters(
          timesteps, batch_size, seed, input_image=input_image_tensor, input_img_noise_t=timestep
        )
        latent = latent_orgin * latent_mask_tensor + latent * (1 - latent_mask_tensor)
      
      if progress_call_back is not None and callable(progress_call_back):
        progress = int((num_steps - index) / num_steps * 100)
        progress_call_back(progress, latent, input_image_array, input_mask_array, self)
    # Decoding stage
    decoded = self.decoder.predict_on_batch(latent)
    decoded = ((decoded + 1) / 2) * 255
    
    if input_mask is not None:
      # Merge inpainting output with original image
      decoded = input_image_array * (1 - input_mask_array) + np.array(decoded) * input_mask_array
    
    return np.clip(decoded, 0, 255).astype("uint8")
  
  def timestep_embedding(self, timesteps, dim=320, max_period=10000):
    half = dim // 2
    freqs = np.exp(
      -math.log(max_period) * np.arange(0, half, dtype="float32") / half
    )
    args = np.array(timesteps) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return tf.convert_to_tensor(embedding.reshape(1, -1), dtype=self.dtype)
  
  def add_noise(self, x, t, noise=None):
    batch_size, w, h = x.shape[0], x.shape[1], x.shape[2]
    if noise is None:
      noise = tf.random.normal((batch_size, w, h, 4), dtype=self.dtype)
    sqrt_alpha_prod = _ALPHAS_CUMPROD[t] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - _ALPHAS_CUMPROD[t]) ** 0.5
    
    return sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise
  
  def get_starting_parameters(self, timesteps, batch_size, seed, input_image=None, input_img_noise_t=None):
    n_h = self.img_height // 8
    n_w = self.img_width // 8
    alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
    alphas_prev = [1.0] + alphas[:-1]
    if input_image is None:
      latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
    else:
      latent = self.encoder(input_image)
      latent = tf.repeat(latent, batch_size, axis=0)
      latent = self.add_noise(latent, input_img_noise_t)
    return latent, alphas, alphas_prev
  
  def get_model_output(
      self,
      latent,
      t,
      context,
      unconditional_context,
      unconditional_guidance_scale,
      batch_size,
  ):
    timesteps = np.array([t])
    t_emb = self.timestep_embedding(timesteps)  # a normal random vector
    t_emb = np.repeat(t_emb, batch_size, axis=0)
    unconditional_latent = self.diffusion_model.predict_on_batch(
      [latent, t_emb, unconditional_context]
    )
    latent = self.diffusion_model.predict_on_batch([latent, t_emb, context])
    return unconditional_latent + unconditional_guidance_scale * (
        latent - unconditional_latent
    )
  
  def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature, seed):
    sigma_t = 0
    sqrt_one_minus_at = math.sqrt(1 - a_t)
    pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)
    
    # Direction pointing to x_t
    dir_xt = math.sqrt(1.0 - a_prev - sigma_t ** 2) * e_t
    noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
    x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
    return x_prev, pred_x0


def get_models(img_height, img_width, download_weights=True):
  n_h = img_height // 8
  n_w = img_width // 8
  
  # Create text encoder
  input_word_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype="int32")
  input_pos_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype="int32")
  embeds = CLIPTextTransformer()([input_word_ids, input_pos_ids])
  text_encoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)
  
  # Creation diffusion UNet
  context = keras.layers.Input((MAX_TEXT_LEN, 768))
  t_emb = keras.layers.Input((320,))
  latent = keras.layers.Input((n_h, n_w, 4))
  unet = UNetModel()
  diffusion_model = keras.models.Model(
    [latent, t_emb, context], unet([latent, t_emb, context])
  )
  
  # Create decoder
  latent = keras.layers.Input((n_h, n_w, 4))
  decoder = Decoder()
  decoder = keras.models.Model(latent, decoder(latent))
  
  inp_img = keras.layers.Input((img_height, img_width, 3))
  encoder = Encoder()
  encoder = keras.models.Model(inp_img, encoder(inp_img))
  
  if download_weights:
    text_encoder_weights_fpath = keras.utils.get_file(
      origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/text_encoder.h5",
      file_hash="d7805118aeb156fc1d39e38a9a082b05501e2af8c8fbdc1753c9cb85212d6619",
    )
    diffusion_model_weights_fpath = keras.utils.get_file(
      origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
      file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
    )
    decoder_weights_fpath = keras.utils.get_file(
      origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
      file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
    )
    
    encoder_weights_fpath = keras.utils.get_file(
      origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
      file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
    )
    
    text_encoder.load_weights(text_encoder_weights_fpath)
    diffusion_model.load_weights(diffusion_model_weights_fpath)
    decoder.load_weights(decoder_weights_fpath)
    encoder.load_weights(encoder_weights_fpath)
  return text_encoder, diffusion_model, decoder, encoder


@lru_cache
def get_or_create_generate(img_height=1000, img_width=1000,
                           jit_compile=True if tf.test.is_gpu_available() else False,
                           download_weights=True):
  print('get_or_create_stable_diffusion')
  return StableDiffusion(img_height, img_width, jit_compile, download_weights)
