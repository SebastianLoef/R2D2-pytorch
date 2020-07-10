
from heapq import heappush, heappop, heapify

import torch
class Node():
    def __init__(self, data):
        self.data = data
        self.size = 0
        self.left = None
        self.Right = None

class SumTree():
    def __init__(self, size):
        self.root = Node(None) 

    def _addLeaf(self, data, node):
        node.size += 1
        if node.Left is None:
            node.left = Node(data)
        elif node.right is None:
            node.right = Node(data)
        else: 
            if node.left.size > node.right.size:
                self._addLeaf(data, node.right)
            else:
                self._addLeaf(data, node.left)

    def addLeaf(self, data):
        self._addLeaf(data, self.root)


        


    



