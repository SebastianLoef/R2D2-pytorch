import time

from itertools import count
import torch.multiprocessing as mp
from collections import deque, namedtuple
import numpy as np

import torch

import model
import equations
from utils import net_update_check, get_latest_nets
from wrappers import make_atari, wrap_deepmind

Package = namedtuple("Package", ["h", "seq"])

QItem = namedtuple("QItem", ["type", "item"])

class Actor(mp.Process):
    def __init__(self, name, env_name, net, target_net,
                 buffer, param_update_freq=400,
                 device="cpu"):
        super().__init__()
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True)
        self.env = env
        self.net = net.to(device)
        self.target_net = target_net.to(device)
        self.device = device
        self.queue = buffer
        self.name = name
        self.m = 80
        self.n = 40
        self.gamma = 0.99
        self.exit = mp.Event()

    def prepare_sequence(self, transitions):
        sequence = list(transitions)
        states, _, rewards = zip(*sequence)
        states = states[5:]
        rewards = rewards + (0,)*5
        delta = [sum(r*self.gamma**i
                     for i, r in enumerate(rewards[j:j+5]))
                 for j in range(self.m)]
        delta = torch.Tensor(delta)
        with torch.no_grad():
            states_v = np.array(states) / 255.0
            states_v = states_v.transpose(0, 3, 1, 2)
            states_v = torch.Tensor(states_v).to(self.device)
            q_values, _ = self.net(states_v)
            actions = q_values.max(1)[1].unsqueeze(-1)
            q_values, _ = self.target_net(states_v)
            q_values = q_values.gather(1, actions).squeeze(-1)
            q_values = q_values.cpu()

        delta[:-5] += equations.h_inv(q_values, 1e-3)
        delta = equations.h(delta, 1e-3)
        p = 0.9*torch.max(delta) + (1 - 0.9)*torch.mean(delta)
        return p, sequence

    def run(self):
        net_iter_id = 0
        transition_buffer = deque(maxlen=self.m)
        hidden_state_buffer = deque(maxlen=self.m)
        sequence_buffer = []
        speed = deque(maxlen=5)
        while not self.exit.is_set():
            transition_buffer.clear()
            sequence_buffer.clear()
            state = self.env.reset()
            start_time = time.time()
            hidden_state = None
            done = False
            if net_update_check(net_iter_id):
                net_name, tgt_net_name = get_latest_nets()
                self.net.load_state_dict(torch.load(net_name))
                self.target_net.load_state_dict(torch.load(tgt_net_name))
            for frame in count():
                with torch.no_grad():
                    state_v = np.array(state) / 255.0
                    state_v = state_v.transpose(2, 0, 1)
                    state_v = torch.Tensor([state_v]).to(self.device)
                    q_values, hidden_state = self.net(state_v)
                    action = q_values.max(1)[1]
                action = action.item()
                new_state, reward, done, _ = self.env.step(action)

                transition_buffer.append([state, action, reward])
                hidden_state = (hidden_state[0].cpu(), hidden_state[1].cpu())
                hidden_state_buffer.append(hidden_state)
                state = new_state
                if frame % self.n == 0:
                    if frame >= self.m:
                        p, prepared_seq = self.prepare_sequence(transition_buffer)
                        package = Package(hidden_state_buffer[0], prepared_seq)
                        sequence_buffer.append([p, package])
                if done:
                    break
            if len(transition_buffer) >= self.m:
                print("Putting into queue")
                p, prepared_seq = self.prepare_sequence(transition_buffer)
                package = Package(hidden_state_buffer[0], prepared_seq)
                sequence_buffer.append([p, package])
                while sequence_buffer:
                    item = QItem("add", sequence_buffer.pop(0))
                    self.queue.put(item)

            time_taken = time.time() - start_time
            speed.append(frame/time_taken)
        
            #print(f"Speed: {np.mean(speed):.2f} f/s")
        return

    def stop(self):
        self.exit.set()


if __name__ == "__main__":
    ENV_NAME = "BreakoutNoFrameskip-v4"
    device = torch.device("cuda:1")
    net = model.R2D2(4).to(device)
    mp.set_start_method('spawn')
    threads = []
    buffer = mp.Queue()
    for i in range(12):
        thread = Actor(f"Thread {i}", ENV_NAME, net,
                       net, buffer, device=device)
        threads.append(thread)

    for thread in threads:
        thread.start()

    time.sleep(20)
    for thread in threads:
        thread.stop()
    time.sleep(5)
    for thread in threads:
        print(f"Child process state: {thread.is_alive()}")
    print(buffer.qsize())
