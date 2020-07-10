import model
import equations
from wrappers import make_atari, wrap_deepmind

import sys
import time
import random
import threading
from itertools import count
from multiprocessing import Process, Manager, set_start_method
import numpy as np
from collections import deque

import torch

class Actor(Process):
    def __init__(self, name, env_name, net, target_net, buffer, param_update_freq=400,
                 device="cpu"):
        super().__init__()
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True)
        self.env = env
        self.net = net.to(device)
        self.target_net = target_net.to(device)
        self.device = device
        self.replaymemory = buffer
        self.name = name
        self.m = 80
        self.n = 40
        self.gamma = 0.99

    def prepare_sequence(self, transitions):
        sequence = list(transitions)
        states, _, rewards = zip(*sequence)
        states = states[5:]
        rewards = rewards + (0,)*5
        delta = [sum(r*self.gamma**i for i, r in
                    enumerate(rewards[j:j+5]))
                for j in range(self.m)]
        delta = torch.Tensor(delta)
        with torch.no_grad():
            states_v = np.array(states) / 255.0
            states_v = states_v.transpose(0, 3, 1, 2)
            states_v = torch.Tensor(states_v).to(self.device)
            q_values = self.net(states_v)
            actions = q_values.max(1)[1].unsqueeze(-1)
            q_values = self.target_net(states_v)
            q_values = q_values.gather(1, actions).squeeze(-1)
            q_values = q_values.cpu()
            
        delta[:-5] += equations.h_inv(q_values, 1e-3)
        delta = equations.h(delta, 1e-3)
        p = 0.9*torch.max(delta) + (1 - 0.9)*torch.mean(delta) 
        return p, sequence

    def run(self):
        transition_buffer = deque(maxlen=self.m)
        sequence_buffer = []
        speed = deque(maxlen=5)
        while True:
            transition_buffer.clear()
            sequence_buffer.clear()
            state = self.env.reset()
            start_time = time.time()
            done = False
            for frame in count():
                with torch.no_grad():
                    state_v = np.array(state) / 255.0
                    state_v = state_v.transpose(2, 0, 1)
                    state_v = torch.Tensor([state_v]).to(self.device)
                    q_values = self.net(state_v)
                    action = q_values.max(1)[1]
                action = action.item()
                new_state, reward, done, _ = self.env.step(action)
                
                transition_buffer.append([state, action, reward])
                state = new_state
                if frame % self.n == 0:
                    if frame >= self.m:
                        p, sequence = self.prepare_sequence(transition_buffer)
                        sequence_buffer.append([p, sequence])
                if done:
                    break
            if len(transition_buffer) >= self.m:
                p, sequence = self.prepare_sequence(transition_buffer)
                sequence_buffer.append([p, sequence])
                self.replaymemory.extend(sequence_buffer)

            time_taken = time.time() - start_time
            speed.append(frame/time_taken)
            print(f"Speed: {np.mean(speed):.2f} f/s")

if __name__=="__main__":
    ENV_NAME = "BreakoutNoFrameskip-v4"
    device = torch.device("cuda:1")
    net = model.R2D2(4).to(device)
    set_start_method('spawn')
    threads = []
    with Manager() as manager:
        buffer = manager.list()
        for i in range(4):
            thread = Actor(f"Thread {i}", ENV_NAME, net, net, buffer, device=device)
            threads.append(thread)

        for thread in threads:
            thread.start()

        time.sleep(5)
        for thread in threads:

            thread.terminate()
            thread.join()

        print(len(buffer))
