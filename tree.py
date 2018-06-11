import os
import numpy as np
from environment import happy


class Node(object):

    def __init__(self, state, action, reward, offspring):
        self.state = state
        self.action = action
        self.reward = reward
        self.offspring = offspring


class P_Tree(object):

    def __init__(self, game):
        self.game = game

    def search(state, depth):
        origin = Node()
