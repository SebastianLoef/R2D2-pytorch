"""
    TODO: add locked leafs functionality and add ability to update p for leafs
"""
import numpy as np


class Node():
    def __init__(self, p=0):
        self.p = p
        self.parent = None
        self.left = None
        self.right = None


class Leaf(Node):
    def __init__(self, p=0, data=None, index=None):
        super().__init__(p)
        self.data = data
        self.index = index


class SumTree():
    def __init__(self, size):
        self.leafs = [Leaf(index=i) for i in range(size)]
        self.crown = None
        self.max_size = size

        def fill_parents(unique_nodes):
            former_node = None
            for node in unique_nodes:
                if former_node is None:
                    node.parent = Node()
                    node.parent.left = node
                elif former_node.parent.right is None:
                    former_node.parent.right = node
                    node.parent = former_node.parent
                else:
                    node.parent = Node()
                    node.parent.left = node
                former_node = node
            new_nodes = set(node.parent for node in unique_nodes)
            if len(new_nodes) > 1:
                fill_parents(new_nodes)
            else:
                self.crown = new_nodes.pop()

        fill_parents(self.leafs)

        def get_root():
            node = self.leafs[0]
            while node.parent is not None:
                node = node.parent
            return node

        self.root = get_root()

    def update_branch(self, node, delta_p):
        node.p += delta_p
        if node.parent is not None:
            self.update_branch(node.parent, delta_p)

    def _find(self, p, node, offset=0):
        if hasattr(node, 'data'):
            return (node.index, node.data)
        if node.left is None:
            return self._find(p, node.right, offset)
        if p >= node.left.p+offset:
            return self._find(p, node.right, node.left.p+offset)
        else:
            return self._find(p, node.left, offset)

    def update_leaf(self, idx, p, data=None):
        delta_p = p - self.leafs[idx].p
        if data:
            self.leafs[idx].data = data
        self.leafs[idx].p = p
        self.update_branch(self.leafs[idx].parent, delta_p)


class ReplayMemory(SumTree):
    def __init__(self, size):
        super().__init__(size)
        self.index = 0
        self.locked_idxs = []
        self.size = 0

    def get_samples(self, sample_size):
        p_total = self.crown.p
        bins = np.linspace(0, p_total, sample_size+1)
        sampled_values = np.random.rand(sample_size)
        sampled_values = (bins[1:] - bins[:-1]) * sampled_values + bins[:-1]
        output = zip(*[self._find(p, self.crown) for p in sampled_values])
        output = list(output)
        self.locked_idxs.extend(output[0])
        return output

    def update_p(self, idx, p):
        self.update_leaf(idx, p)
        self.locked_idxs.remove(idx)

    def append(self, p, data):
        while self.index in self.locked_idxs:
            self.index = (self.index + 1) % self.max_size
        self.update_leaf(self.index, p, data)
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def extend(self, sequence_batch):
        for sequence in sequence_batch:
            p, data = sequence
            self.append(p, data)
