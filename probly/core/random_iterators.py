import numpy as np


class RandomIterator:
    def __init__(self, seed=1):
        self.current_seed = seed
        self.random_state = np.random.RandomState(seed).get_state()[1]
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.random_state):
            self.current_index += 1
            self.random_state = np.random.RandomState(self.current_seed).get_state()[1]
            self.current_index = 0
        result = self.random_state[self.current_index]
        self.current_index += 1
        return result
