from absl import logging
import json
import numpy as np
from skimage import morphology
from scipy import ndimage

import threading
import time
import traceback
from typing import Dict, Any
from dash.long_callback.managers import BaseLongCallbackManager
from dash.exceptions import PreventUpdate
from dash_canvas.utils.parse_json import _indices_of_path

_pending_value = "__$pending__"


class MemCache(object):
  def __init__(self, jobs):
    self._jobs = jobs
    self._data: Dict[str, Any] = {}
    self._key_to_job: Dict[str, int] = {}
    self._lock = threading.RLock()
    self._lock_key_to_job = threading.RLock()
  
  def transact(self):
    return self._lock
  
  def delete(self, key):
    if key is None:
      return
    with self._lock:
      if key in self._data:
        del self._data[key]
      elif f'{key}-progress' in self._data:
        del self._data[f'{key}-progress']
      else:
        key = key.split('-')[0]
        if key in self._data:
          del self._data[key]
    
    with self._lock_key_to_job:
      if key in self._key_to_job:
        del self._key_to_job[key]
      elif f'{key}-progress' in self._key_to_job:
        del self._key_to_job[f'{key}-progress']
      else:
        key = key.split('-')[0]
        if key in self._key_to_job:
          del self._key_to_job[key]
  
  def get(self, key, default=BaseLongCallbackManager.UNDEFINED, timeout=1):
    if key is None:
      return default
    with self._lock:
      if key in self._data:
        return self._data.get(key, default)
    
    try:
      job = self.get_job(key)
      if job in self._jobs:
        thread = self._jobs[job]
        cnt = 0
        while thread.is_alive() and cnt < timeout * 5:
          time.sleep(0.2)
          cnt += 1
          with self._lock:
            if key in self._data:
              return self._data.get(key, default)
      with self._lock:
        return self._data.get(key, default)
    except Exception as e:
      logging.info(e)
      with self._lock:
        return self._data.get(key, default)
  
  def set(self, key: str, value: Any):
    if key is None:
      return
    with self._lock:
      self._data[key] = value
  
  def touch(self, key, expire):
    if key is None:
      return
    with self._lock:
      self._data[key] = BaseLongCallbackManager.UNDEFINED
  
  def add_key_and_job(self, key, job):
    with self._lock_key_to_job:
      self._key_to_job[key] = job
  
  def get_job(self, key: str):
    with self._lock_key_to_job:
      if key in self._key_to_job:
        return self._key_to_job[key]
      elif f"{key}-progress" in self._key_to_job:
        return self._key_to_job[f"{key}-progress"]
      else:
        key = key.split('-')[0]
        return self._key_to_job.get(key, -1)


class LocalManager(BaseLongCallbackManager):
  
  def __init__(self, cache_by=None, expire=None):
    self.jobs: Dict[int, threading.Thread] = {}
    self.handle: MemCache = MemCache(self.jobs)
    self.expire = expire
    super(LocalManager, self).__init__(cache_by)
  
  def terminate_job(self, job: int):
    job = int(job)
    if job in self.jobs:
      if self.jobs[job].is_alive():
        self.jobs[job].join()
      try:
        with self.handle.transact():
          del self.jobs[job]
      except Exception as e:
        logging.info(e)
    return True
  
  def terminate_unhealthy_job(self, job):
    return self.terminate_job(job)
  
  def job_running(self, job):
    job = int(job)
    if job in self.jobs:
      return self.jobs[job].is_alive()
    return False
  
  def make_job_fn(self, fn, progress: bool):
    return _make_job_fn(fn, self.handle, progress)
  
  def clear_cache_entry(self, key):
    self.handle.delete(key)
  
  def call_job_fn(self, key, job_fn, args, context):
    from threading import Thread
    
    thread = Thread(target=job_fn, args=(key, self._make_progress_key(key), args))
    thread.start()
    job = thread.native_id
    self.handle.add_key_and_job(key, job)
    with self.handle.transact():
      self.jobs[job] = thread
    return job
  
  def get_progress(self, key):
    progress_key = self._make_progress_key(key)
    return self.handle.get(progress_key)
  
  def result_ready(self, key):
    return self.handle.get(key) is not None
  
  def get_result(self, key: str, job: int):
    # Get result value
    result = self.handle.get(key, self.UNDEFINED)
    if result is self.UNDEFINED:
      return self.UNDEFINED
    
    # Clear result if not caching
    if self.cache_by is None:
      self.clear_cache_entry(key)
    else:
      if self.expire:
        self.handle.touch(key, expire=self.expire)
    
    self.clear_cache_entry(self._make_progress_key(key))
    
    if job:
      self.terminate_job(job)
    return result


def _make_job_fn(fn, cache, progress):
  def job_fn(result_key, progress_key, user_callback_args):
    def _set_progress(progress_value):
      if not isinstance(progress_value, (list, tuple)):
        progress_value = [progress_value]
      
      job = cache.get_job(progress_key)
      cache.add_key_and_job(progress_key, job)
      cache.set(progress_key, progress_value)
    
    maybe_progress = [_set_progress] if progress else []
    
    try:
      if isinstance(user_callback_args, dict):
        user_callback_output = fn(*maybe_progress, **user_callback_args)
      elif isinstance(user_callback_args, (list, tuple)):
        user_callback_output = fn(*maybe_progress, *user_callback_args)
      else:
        user_callback_output = fn(*maybe_progress, user_callback_args)
    except PreventUpdate:
      cache.set(result_key, {"_dash_no_update": "_dash_no_update"})
    except Exception as err:  # pylint: disable=broad-except
      cache.set(
        result_key,
        {
          "long_callback_error": {
            "msg": str(err),
            "tb": traceback.format_exc(),
          }
        },
      )
    else:
      cache.set(result_key, user_callback_output)
  
  return job_fn


def parse_jsonstring(string, shape=None, scale=1):
  if shape is None:
    shape = (500, 500)
  mask = np.zeros(shape, dtype=np.uint8)
  try:
    data = json.loads(string)
  except:
    return mask
  scale = 1
  for obj in data['objects']:
    if obj['type'] == 'image':
      scale = obj['scaleX']
    elif obj['type'] == 'path':
      scale_obj = obj['scaleX']
      inds = _indices_of_path(obj['path'], scale=scale / scale_obj)
      radius = round(obj['strokeWidth'] / 2. / scale)
      mask_bool = np.zeros(shape, dtype=np.bool)
      mask_bool[inds[0], inds[1]] = 1
      mask_bool = ndimage.binary_dilation(mask_bool,
                                          morphology.disk(radius))
      
      alpha = float(obj['stroke'].strip('{}()').split(',')[-1])
      mask_int8 = np.zeros(shape, dtype=np.uint8)
      mask_int8[mask_bool] = int(alpha * 255)
      mask = np.maximum(mask_int8, mask)
  return mask
