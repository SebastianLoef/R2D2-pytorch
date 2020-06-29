import model
from wrappers import make_atari, wrap_deepmind

import sys
import time
import random
import threading
from multiprocessing import Process, Manager, set_start_method
import numpy as np
from collections import deque

import torch

class Actor(Process):
    def __init__(self, name, env_name, net, buffer, param_update_freq=400,
                 device="cpu"):
        super().__init__()
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True)
        self. env = env
        self.network = net.to(device)
        self.device = device
        self.buffer = buffer
        self.local_buffer = []
        self.name = name
        self.speed = deque(maxlen=5)

    def run(self):
        while True:
            state = self.env.reset()
            done = False
            start_time = time.time()
            frame_count = 0
            while not done:
                with torch.no_grad():
                    state_v = np.array(state)/255.0
                    state_v = state_v.transpose(2, 0, 1)
                    state_v = torch.Tensor([state_v]).to(self.device)
                    q_values = self.network(state_v)
                    action = q_values.max(1)[1]
                action = action.item()

                new_state, reward, done, _ = self.env.step(action)
                self.local_buffer.append([state, action, reward])
                state = new_state
                frame_count += 1
            time_taken = time.time() - start_time
            self.speed.append(frame_count/time_taken)
            self.buffer.extend(self.local_buffer)
            self.local_buffer = []

if __name__=="__main__":
    ENV_NAME = "BreakoutNoFrameskip-v4"
    device = torch.device("cuda:1")
    net = model.R2D2(4).to(device)
    set_start_method('spawn')
    threads = []
    with Manager() as manager:
        buffer = manager.list()
        for i in range(14):
            thread = Actor(f"Thread {i}", ENV_NAME, net, buffer, device=device)
            threads.append(thread)

        for thread in threads:
            thread.start()

        time.sleep(100)
        for thread in threads:

            thread.terminate()
            thread.join()

        print(len(buffer))
