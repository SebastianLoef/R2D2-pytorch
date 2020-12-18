from actor import Actor
from replaymemory import ReplayMemory
import model
from collections import namedtuple
import time
import torch.multiprocessing as mp
import torch

QItem = namedtuple("QItem", ["type", "item"])


class ReplayCommunicator(mp.Process):
    def __init__(self, q, loader):
        super().__init__()
        self.q = q
        self.loader = loader
        self.replaymemory = ReplayMemory(1000)
        self.exit_call = mp.Event()

    def add(self, item):
        print(f"ReplayCommunicator added item to Replaymemory at {time.time()}")
        p = item[0]
        self.replaymemory.append(p, item[1])

    def sample_batch(self):
        print(f"sampled batch at {time.time()}")
        self.loader.put(self.replaymemory.get_samples(64))

    def update_p_values(self, obj):
        del obj


    def run(self):
        while not self.exit_call.is_set():
            while not self.q.empty():
                print(f"Current q size: {self.q.qsize()}")
                next_task = self.q.get()
                if next_task.type == 'append':
                    self.add(next_task.item)
                elif next_task.type == 'sample_batch':
                    self.sample_batch()
                elif next_task.type == 'update_p_values':
                    self.update_p_values(next_task.item)
                else:
                    print("ReplayCommunicator has a unidentifed task type.")
            time.sleep(0.2)
            print(f"Replay memory size {self.replaymemory.size}")

    def stop(self):
        self.exit_call.set()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    mp.set_start_method("spawn")
    queue = mp.Queue(100)
    loader = mp.Queue()

    ENV_NAME = "BreakoutNoFrameskip-v4"
    device = torch.device("cuda:1")
    net = model.R2D2(4).to(device)
    # Creates and starts all actors
    actors = []
    for i in range(8):
        actor = Actor(f"Thread {i}", ENV_NAME, net,
                      net, queue, device=device)
        actor.start()
        actors.append(actor)
    repcom = ReplayCommunicator(queue, loader)
    repcom.start()
    load_protocol = QItem("sample_batch", None)
    queue.put(load_protocol)
    queue.put(load_protocol)
    for step in range(100):
        queue.put(load_protocol)
        time.sleep(1)
        print(loader.qsize())
        l = loader.get()
        queue.put(QItem("update_p_values", l))
        if step % 2500 == 0:
            # update actors
            pass

    # Stops all actors
    for actor in actors:
        actor.stop()
    repcom.stop()
