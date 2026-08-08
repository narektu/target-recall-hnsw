import math
import random

class LevelSampler:
    def __init__(self, M: int, seed: int | None = None):
        self.m_l = 1.0 / math.log(M)
        self._rng = random.Random(seed)

    def sample(self) -> int:
        return math.floor(-math.log(self._rng.random()) * self.m_l)
