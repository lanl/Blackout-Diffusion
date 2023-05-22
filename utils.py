import torch
import tensorflow as tf
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    state['lossHistory'] = loaded_state['lossHistory']
    return state

def restore_checkpoint_withEval(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    state['lossHistory'] = loaded_state['lossHistory']
    state['evalLossHistory'] = loaded_state['evalLossHistory']
    return state

def restore_checkpoint_BD(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model_br'].load_state_dict(loaded_state['model_br'], strict=False)
    state['model_dr'].load_state_dict(loaded_state['model_dr'], strict=False)
    state['ema_br'].load_state_dict(loaded_state['ema_br'])
    state['ema_dr'].load_state_dict(loaded_state['ema_dr'])
    state['step'] = loaded_state['step']
    state['lossHistory'] = loaded_state['lossHistory']
    return state

def restore_checkpoint_RW(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model_ir'].load_state_dict(loaded_state['model_br'], strict=False)
    state['model_dr'].load_state_dict(loaded_state['model_dr'], strict=False)
    state['ema_ir'].load_state_dict(loaded_state['ema_br'])
    state['ema_dr'].load_state_dict(loaded_state['ema_dr'])
    state['step'] = loaded_state['step']
    state['lossHistory'] = loaded_state['lossHistory']
    state['evalLossHistory'] = loaded_state['evalLossHistory']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'lossHistory': state['lossHistory']
  }
  torch.save(saved_state, ckpt_dir)
 

def save_checkpoint_withEval(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'lossHistory': state['lossHistory'],
    'evalLossHistory': state['evalLossHistory']
  }
  torch.save(saved_state, ckpt_dir)
    
def save_checkpoint_BD(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model_br': state['model_br'].state_dict(),
    'model_dr': state['model_dr'].state_dict(),
    'ema_br': state['ema_br'].state_dict(),
    'ema_dr': state['ema_dr'].state_dict(),
    'step': state['step'],
    'lossHistory': state['lossHistory']  
  }
  torch.save(saved_state, ckpt_dir)
    
def save_checkpoint_RW(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model_br': state['model_ir'].state_dict(),
    'model_dr': state['model_dr'].state_dict(),
    'ema_br': state['ema_ir'].state_dict(),
    'ema_dr': state['ema_dr'].state_dict(),
    'step': state['step'],
    'lossHistory': state['lossHistory'],
    'evalLossHistory': state['evalLossHistory']
  }
  torch.save(saved_state, ckpt_dir)