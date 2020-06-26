from wrappers import make_atari, wrap_deepmind

import numpy as np
import random

import torch

class Actor:
    def __init__(self, env_name, net, B, param_update_freq=400):
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True)
        self. env = env
        self.network = net
        self.B = B

    
    def play(self):
        state = self.env.reset()

