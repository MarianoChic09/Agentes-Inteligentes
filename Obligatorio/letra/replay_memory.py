import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)

class ReplayMemory:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, done, next_state):
      self.memory.append(Transition(state, action, reward, done, next_state))

    def sample(self, batch_size):
      return random.sample(self.memory, batch_size)

    def __len__(self):
      return len(self.memory)
