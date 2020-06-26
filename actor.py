import model
from wrappers import make_atari, wrap_deepmind

import sys
import time
import random
import threading
import numpy as np
from collections import deque

import torch

class Actor(threading.Thread):
    def __init__(self, name, env_name, net, buffer, param_update_freq=400):
        threading.Thread.__init__(self)
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True)
        self. env = env
        self.network = net
        self.buffer = buffer
        self.speed = deque(maxlen=5)
        self._stop_event = threading.Event()
    
    def run(self):
        while True:
            state = self.env.reset()
            done = False
            start_time = time.time()
            frame_count = 0
            while not done:
                state_v = np.array(state)/255.0
                state_v = state_v.transpose(2, 0, 1)
                state_v = torch.Tensor([state_v])
                q_values = self.network(state_v)
                action = q_values.max(1)[1]
                action = action.item()

                new_state, reward, done, _ = self.env.step(action)
                self.buffer.append([state, action, reward])
                state = new_state
                frame_count += 1
            time_taken = time.time() - start_time
            self.speed.append(frame_count/time_taken)
            print(f"thread {self.name} speed: {np.mean(self.speed):.1f} f/s")
            if self.stopped():
                break
    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
if __name__=="__main__":
    ENV_NAME = "BreakoutNoFrameskip-v4"
    net = model.R2D2(4)
    
    threads = []
    buffer = []
    for i in range(2):
        thread = Actor(f"Thread {i}", ENV_NAME, net, buffer)
        threads.append(thread)

    for thread in threads:
        thread.start()

    time.sleep(10)
    for thread in threads:
        thread.stop()
        thread.join()

            

