import random
from heapq import heappush, heappop, heapify

import torch
class Node():
    def __init__(self, p=0, data=None):
        self.p = p
        self.data = data
        self.size = 0
        self.parent = None
        self.left = None
        self.right = None


class SumTree():
    def __init__(self, size):
        self.leafs = [Node() for _ in range(size)]
        self.max_size = size
        self.index = 0

        def fill_parents(unique_nodes):
            former_node = None
            for node in unique_nodes:
                if former_node is None:
                    node.parent = Node()
                    node.parent.left = node
                    former_node = node
                    continue

                if former_node.parent.right is None:
                    former_node.parent.right = node
                    node.parent = former_node.parent
                else:
                    node.parent = Node()
                    node.parent.left = node
                former_node = node
            new_nodes = set(node.parent for node in unique_nodes)
            if len(new_nodes) > 1:
                fill_parents(new_nodes)

        fill_parents(self.leafs)

        def get_root():
            node = self.leafs[0]
            while node.parent is not None:
                node = node.parent
            return node

        self.root = get_root()

    def update_branch(self, node, dp):
        node.p += dp 
        if node.parent is not None:
            self.update_branch(node.parent, dp)

    def append(self, p, data):
        diff = p - self.leafs[self.index].p
        self.leafs[self.index].data = data
        self.leafs[self.index].p = p
        self.update_branch(self.leafs[self.index].parent, diff)

        self.index = (self.index + 1) % self.max_size

    def extend(self, sequence_batch):
        for sequence in sequence_batch:
            p, data = sequence
            self.append(p, data)

    def get_batch(self, batch_size):
        pass



    
        


    



