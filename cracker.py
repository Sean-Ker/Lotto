#%%
import math
import random
import time
from itertools import permutations

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import collections
import random
from joblib import Parallel, delayed
from tqdm import tqdm
from prng import MT19937, LCG

# constants
game = "dgrad"
NUM_LIM = (1,49)
DGRAD_LIM = (1,7)
TESTING_IND = 0.9

df = pd.read_csv("data/daily_grand/DailyGrand.csv", index_col=0)
df.reset_index(drop=True, inplace=True)
df.drop(['DRAW NUMBER','DRAW DATE','PRIZE DIVISION', 'SEQUENCE NUMBER'], axis=1, inplace=True)
df.columns = ['n1', 'n2', 'n3', 'n4', 'n5', 'ngrand']
df_no_dgrand = df.drop('ngrand', axis=1)
assert not df_no_dgrand.duplicated().any()

n_train = int(len(df)*TESTING_IND)

df_train = df.iloc[:n_train]
df_test =df.iloc[n_train:]
assert len(df_train) + len(df_test) == len(df)

def create_groups(df):
    data = df.values.astype(np.int)
    groups = []
    all_nums = []
    all_grands = []
    for row in data:
        groups.append(row[:-1])
        all_nums += list(row[:-1])
        if row[-1] != 0:
            groups.append(int(row[-1]))
            all_grands.append(int(row[-1]))
    return groups, all_nums, all_grands

groups_y, all_nums, all_grands = create_groups(df_train)
groups_test, all_nums_test, all_grands_test = create_groups(df_test)

def create_freq(iterratable):
    counter = collections.Counter(iterratable)
    return {k:counter[k] for k in sorted(counter)}
    # list(counter.values())

nums_freq_y = create_freq(all_nums)
grands_freq_y = create_freq(all_grands)
# numbers_dist =
# ngrand_dist =

#### Idea: don't do code abbove, get flatten list of winning numbers and the grand numbers and get their distributio.

def mse(lst1, lst2):
    assert len(lst1) == len(lst2)
    if type(lst1)==dict:
        if not type(lst2)==dict:
            raise ValueError
        return sum([(lst1[k] - lst2[k])**2 for k in lst1.keys()]) / len(lst1)
    return np.float32(sum([(x[0] - x[1])**2 for x in zip(lst1, lst2)]) / len(lst1))

def scale_01(lst, lmin=None, lmax=None):
    if type(lst)==dict:
        lst_min = lmin if lmin else min(lst.values())
        lst_max = lmax if lmax else max(lst.values())
        return {k:(v-lst_min)/(lst_max - lst_min) for k,v in lst.items()}
    lst_min = lmin if lmin else min(lst)
    lst_max = lmax if lmax else max(lst)
    return [(x-lst_min)/(lst_max - lst_min) for x in lst]

def mse_01(lst1, lst2):
    return mse(scale_01(lst1), scale_01(lst2))

def mae(lst1, lst2):
    assert len(lst1) == len(lst2)
    return np.float32(sum([abs(x[0] - x[1]) for x in zip(lst1, lst2)]) / len(lst1))

# a= mt_rng.integer((1,49),100)
# b = mt_rng.integer((1,49),100)
# mse(a,b)
# mse_01(a,b)

# %%

def simulate_draws(rng_algo, seed_range, return_groups=False):
    all_res = {}
    start = time.time()

    for seed in seed_range:
        res = simulate_draw(seed, rng_algo, return_groups)
        if res is None:
            continue
        all_res[seed] = res

    if return_groups:
        return all_res

    print(f"time took: {time.time()-start} seconds")

    res_df = pd.DataFrame.from_dict(all_res).transpose()
    res_df.columns=['initial_n','num_score','dgrad_score']
    res_df['average']=(res_df['num_score']+res_df['dgrad_score'])/2
    res_df.sort_values('average')
    return res_df

def simulate_draw(seed, rng_algo, return_groups=False):
    rng = rng_algo(seed=seed)
    res = []
    pred_nums = []
    pred_dgrand = []

    for i in range(16):
        generated = rng.integer((1,49),1)
        if generated in groups_y[0]:
            res.append(i)
            pred_nums.append(generated)
            break
    else:
        return None
    pred_nums += rng.integer((1,49),4)

    for i in groups_y[1:]:
        if isinstance(i, int):
            pred_dgrand.append(rng.integer((1,7),1))
        else:
            pred_nums+=(rng.integer((1,49),5))

    assert len(pred_nums) == len(all_nums)
    assert len(pred_dgrand) == len(all_grands)

    pred_nums_freq = create_freq(pred_nums)
    pred_grands_freq = create_freq(pred_dgrand)

    res.append(mse(pred_nums_freq, nums_freq_y))
    res.append(mse(pred_grands_freq, grands_freq_y))
    if return_groups:
        res.append([pred_nums_freq, pred_grands_freq])
    return {seed: res}

class SampleGen:
    def __init__(self, population):
        self.population = set(population)
    def generate(self, n):
        n = min(n, len(self.population))
        generated = random.sample(self.population, n)
        self.population = self.population - set(generated)
        return generated

class RangeGen:
    def __init__(self, original_range):
        self.current_batch = original_range
    def batch(self, n):
        start = self.current_batch.start; stop = self.current_batch.stop
        if start == stop:
            return None
        n = min(n, stop - start)
        self.current_batch = range(start + n, stop)
        return range(start, start+n)


# res_df = simulate_draws(MT19937,range(0,4000))
# best_seed = simulate_draws(MT19937,range(1915,1916),return_groups=True)

# start_time = time.perf_counter()

range_gen = RangeGen(range(2300, 2**32-1))
best_seed = 300
best_so_far = pd.Series(list(simulate_draw(best_seed, MT19937).values())[0],dtype=float,
            index=['initial_n','num_score','dgrad_score'], name=best_seed)
best_so_far['average'] = (best_so_far.num_score + best_so_far.dgrad_score)/2

# while True:
#     current_batch = range_gen.batch(5000)
#     print(f"current batch: range{current_batch.start, current_batch.stop}")
#     list_of_dict = Parallel(n_jobs=-1, verbose=1)(delayed(simulate_draw)(seed, MT19937) for seed in current_batch)
#     list_of_dict = [x for x in list_of_dict if x is not None]
#     all_res = dict((key,d[key]) for d in list_of_dict for key in d)
#     res_df = pd.DataFrame.from_dict(all_res).transpose()
#     res_df.columns=['initial_n','num_score','dgrad_score']
#     res_df['average']=(res_df['num_score']+res_df['dgrad_score'])/2
#     res_df.sort_values('average')
#     # print(f'took {time.perf_counter()-start_time} seconds.')
#     batch_best = res_df.iloc[0]
#     if best_so_far is None:
#         best_so_far = batch_best
#     if best_so_far.average > batch_best.average:
#         best_so_far = batch_best
#         print(batch_best)
#     if batch_best.average < 40:
#         print("found less than 40 average!")
#         break

# [x for x in res if x==1 or x==49]

#%%

def perfect_match(seed, rng_algo, groups):
    rng = rng_algo(seed=seed)
    draw_record = 0
    for start_ind in range(16):
        generated = rng.integer((1,49),1)
        if generated in groups[0]:
            draw_record +=1
            groups[0] = np.delete(groups[0], np.where(groups[0]==generated))
            break
    else:
        return [seed, 0]

    for cg in groups:
        if isinstance(cg, int):
            if cg == rng.integer((1,7),1):
                draw_record+=1
            else: return [seed, draw_record]
        else:
            for _ in range(len(cg)):
                generated = rng.integer((1,49), 1)
                if generated in cg:
                    draw_record+=1
                else: return [seed, draw_record]
    return [seed, draw_record]


# groups = groups_y.copy()
start_time = time.perf_counter()
n=2**25
res = []
for seed in tqdm(random.sample(range(10**9-1, 2**32), n)):
    res.append(perfect_match(seed, LCG, groups_y.copy()))
# res = Parallel(n_jobs=-1, verbose=10)(delayed(perfect_match)(seed, MT19937, groups_y.copy()) for seed in range(n))

res = sorted(res, key=lambda x: x[1], reverse=True)
res_dict = dict(res)
print(f'took {round(time.perf_counter()-start_time, 2)} seconds.')
print({k: res_dict[k] for k in list(res_dict)[:10]})
# perfect_match(2136656900, LCG, groups_y.copy())
# %%
lcg = LCG(2136656900)
lcg.integer((1,49),5)
lcg.integer((1,49),5)
lcg.integer((1,7),1)
lcg.integer((1,49),5)
