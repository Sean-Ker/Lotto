import prng
import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from itertools import permutations
import time

# constants
seed = 1

lcg_sample = []
state_test = (2,3,49) # (a, b, mod)
x = seed
for i in range(20):
    x = ((state_test[0]*x+state_test[1]) % state_test[2])
    lcg_sample.append(x)

# print(res)

def EEA(a: int, b: int):
    """return (g, x, y) such that a*x + b*y = d = gcd(a, b)"""
    x0, x1, y0, y1 = 0, 1, 1, 0
    while a != 0:
        (q, a), b = divmod(b, a), a
        y0, y1 = y1, y0 - q * y1
        x0, x1 = x1, x0 - q * x1
    return b, x0, y0                # (d, x, y)

# def EEA_list_rec(lst):
#     def eea_lst(lst, current):
#         if not lst:
#             return current
#         else:
#             return eea_lst(lst[1:], EEA(lst[0], current)[0])
#     return eea_lst(lst[1:], lst[0])

def EEA_list(lst):
    # print(lst)
    m = lst[0]
    for i in lst[1:]:
        m=math.gcd(m, i)
    return m

class SolvingException(Exception):
    pass

def modinv(a, m):
    g, x, _ = EEA(a, m)
    if g != 1:
        raise SolvingException('modular inverse does not exist')
    else:
        return x % m

def lcg_solve(Xn):
    '''Breaking Linear Congruential Generators'''

    Yn = [Xn[k]-Xn[k-1] for k in range(1, len(Xn))]
    Zn = [abs(Yn[k]*Yn[k-2] - Yn[k-1]**2) for k in range(2, len(Yn))]

    m = EEA_list(Zn)

    Y2_inv = modinv(Yn[1] + m, m)
    Y3 = Yn[2]
    a = (Y2_inv * Y3)%m
    b = (Xn[1] - a *  Xn[0])%m

    x = Xn[0]
    for i in Xn:
        if x != i:
            raise SolvingException('Failed to solve')
        x = (a*x + b)%m
    return (a,b,m)   # x_n+1, (a, b, m)

def lcg_next(seed, state):
    return (seed * state[0] + state[1]) % state[2]

# Tests:
test = [720555190, 133143292, 350469176, 715002068, 822810950, 400865843, 226553034, 200183345]
test_state_solved = lcg_solve(test)
assert test_state_solved == (1664525, 13904216, 1000000007)
assert lcg_next(test[-1], test_state_solved) == 193907871
assert lcg_solve(lcg_sample[:-1]) == state_test
assert lcg_next(lcg_sample[-2], state_test) == lcg_sample[-1]
