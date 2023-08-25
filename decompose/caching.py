import hashlib
from functools import lru_cache, wraps

import numpy as np

"""
Based on https://stackoverflow.com/a/76483281/156884
"""


class HashingWrapper:
    def __init__(self, x: np.array) -> None:
        self.values = x
        # here you can use your own hashing function
        self.h = hashlib.sha224(x.view()).hexdigest()

    def __hash__(self) -> int:
        return hash(self.h)

    def __eq__(self, __value: object) -> bool:
        return __value.h == self.h


def memoizer(expensive_function):
    @lru_cache()
    def cached_wrapper(shell):
        return expensive_function(shell.values)

    @wraps(expensive_function)
    def wrapper(x: np.array):
        shell = HashingWrapper(x)
        return cached_wrapper(shell)

    return wrapper

#     def expensive_function(x: np.array)-> float:
#         time.sleep(0.1)
#         return np.sqrt(x.clip(1).sum())
# a = np.random.random((1, 5))
# a = np.vstack((a,) * 200)
#
# memoized_expensive_function = memoizer(expensive_function)
#
# np.apply_along_axis(expensive_function, 1, a)  # takes 20 seconds
# np.apply_along_axis(memoized_expensive_function, 1, a)  # takes 0.2 seconds