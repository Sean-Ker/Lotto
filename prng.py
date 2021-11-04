from warnings import warn
import collections
import math
import random
import time
from itertools import permutations

import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt

class RNG(object):
    '''A Random Number Generator class to be inherited from'''
    def __init__(self, seed, max_val, float_size):
        self.seed = seed
        self.max_val = max_val
        self.float_size = np.float32 if not float_size else float_size

    def float64(self):
        return

    def integer(self, interval=None, count=1):
        """
        Return a pseudorandom integer. Default is one integer number in {0,1}. Pass in a tuple or list, (a,b), to
        return an integer number on [a,b]. If count is 1, a single number is returned, otherwise a list of numbers
        is returned.
        """
        results = []
        for _ in range(count):
            generated = self.float01()
            if interval is not None:
                results.append(
                    int((interval[1] - interval[0] + 1) * generated + interval[0]))
            else:
                if generated < 0.50:
                    results.append(0)
                else:
                    results.append(1)
        if count == 1:
            return results.pop()
        return results

    def floating(self, interval=None, count=1):
        """
        Return a pseudorandom float. Default is one floating- point number between zero and one. Pass in a tuple or
        list, (a,b), to return a floating-point number on [a,b]. If count is 1, a single number is returned,
        otherwise a list of numbers is returned.
        """
        results = []
        for _ in range(count):
            generated = self.float01()
            if interval is not None:
                results.append(
                    (interval[1] - interval[0]) * generated + interval[0])
            else:
                results.append(generated)
        if count == 1:
            return results.pop()
        else:
            return results


class MT19937(RNG):
    """mt19937"""
    def __init__(self, seed, float_size= None):
        super().__init__(seed, 2**32, float_size)
        self.state = [0]*624
        self.f = 1812433253
        self.m = 397
        self.u = 11
        self.s = 7
        self.b = 0x9D2C5680
        self.t = 15
        self.c = 0xEFC60000
        self.l = 18
        self.index = 624
        self.lower_mask = (1<<31)-1
        self.upper_mask = 1<<31

        # update state
        self.state[0] = seed
        for i in range(1,624):
            self.state[i] = self.__int_32(self.f*(self.state[i-1]^(self.state[i-1]>>30)) + i)

    def __twist(self):
        for i in range(624):
            temp = self.__int_32((self.state[i]&self.upper_mask)+(self.state[(i+1)%624]&self.lower_mask))
            temp_shift = temp>>1
            if temp%2 != 0:
                temp_shift = temp_shift^0x9908b0df
            self.state[i] = self.state[(i+self.m)%624]^temp_shift
        self.index = 0

    def __int_32(self, number):
        return int(0xFFFFFFFF & number)

    def float01(self):
        if self.index >= 624:
            self.__twist()
        y = self.state[self.index]
        y = y^(y>>self.u)
        y = y^((y<<self.s)&self.b)
        y = y^((y<<self.t)&self.c)
        y = y^(y>>self.l)
        self.index+=1
        return self.float_size(y/self.max_val)


class LCG(RNG):
    """LCG Pseudorandom number generater"""

    def __init__(self, seed, modulo=(2 ** 31 - 1), multiplier = 16708, adder = 1, float_size=None):
        """
        Initialize pseudorandom number generator. Accepts an integer or floating-point seed, which is used in
        conjunction with an integer multiplier, k, and the Mersenne prime, j, to "twist" pseudorandom numbers out
        of the latter. This member also initializes the order of the generator's period, so that members floating and
        integer can emit a warning when generation is about to cycle and thus become not so pseudorandom.
        """
        super().__init__(seed = seed, max_val = modulo, float_size = float_size)
        self.current = self.seed = seed
        self.mod = modulo
        self.a = multiplier
        self.b = adder
        self.period = 2**int(math.log(modulo, 2))
        return

    def float01(self):
        self.current = (self.a * self.current + self.b) % self.mod
        self.period -= 1
        if self.period == 0:
            warn("Pseudorandom period nearing!!", category=ResourceWarning)
        return self.float_size(self.current / self.max_val)


# from PIL import Image

# rng = LCG(5167)
# size = 512

# res = rng.floating(count=size**2)
# res = np.array(res).reshape((size, size)) * 255

# Image.fromarray(res,mode="L")


# # %%
# a= np.array((1,0,1))
# scaler = np.array((255,255,255)).reshape(1,1,3)
# a*scaler

#%%
mt_rng = MT19937(2,float_size=np.float64)
mt_rng.floating(None,10)
