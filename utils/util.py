#%%
class DirClass:
    # def _add_ending_if_missing(self, string, key='\\'):
    #     if str(string)[-len(key):] != key:
    #         return string + key
    #     return string

    def _remove_ending_if_present(self, string, key = '\\', loop = True):
        while True:
            if str(string)[-len(key):] == key:
                string = string[:-len(key)]
            else:
                break
            if not loop:
                break

        return string

    mode = '\\'
    secondary = '/'

    def __init__(self, path, relative = False):

        if self.secondary in path:
            path.replace(self.secondary, self.mode)

        self._remove_ending_if_present(path, self.mode)

        if relative:
            if 'DirClass' in str(type(relative)):
                relative = relative.path
            relative = self._remove_ending_if_present(relative, self.mode)
            self.path = relative + self.mode + path + self.mode
            self.name = path
            self.parent = relative.split(self.mode)[-1]
        else:
            self.name = path.split(self.mode)[-1]
            self.path = path + self.mode
            self.parent = path.split(self.mode)[-2]

    def __repr__(self):
        return self.path
    def __str__(self):
        return self.path

    def relative_to(self, other):
        return self._remove_ending_if_present(self.path.replace(other.path, ''), self.mode)

class DirInfo:
    LOTTO = DirClass('C:\\_Workspace\\lotto')
    DATA = DirClass('data', LOTTO)
    UTILS = DirClass('utils\\', LOTTO)
    PLOTS = DirClass('plots\\', LOTTO)

'''######################## Basic Utils ######################## '''
# import os, sys; os.chdir('C:/_Workspace/lotto'); sys.path.append('C:/_Workspace/lotto')
import pickle
from time import time

import numpy as np
import pandas as pd
import random
from datetime import datetime
# from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# from scipy.stats import ks_2samp
np.random.seed(1)
random.seed(1)
# tf.random.set_seed(1)

# minutes -> fsb pro file namee | interval -> string | time_frame -> milliseconds of interval
MINUTES = [1, 5, 15, 30, 60, 240, 1440, 10080]
INTERVALS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
MIN_TO_INTERVAL = {1: '1m', 5: '5m', 15: '15m', 30: '30m',
                   60: '1h', 240: '4h', 1440: '1d', 10080: '1w'}

import logging
# for handler in logging.root.handlers[:]:
# #     logging.root.removeHandler(handler)
#     handler.close()
logging.basicConfig(filename = 'data\debug_logs.log', format='%(asctime)s [%(levelname)s:%(name)s] - %(message)s',level=logging.INFO)
logging.info('################################## Loading Utils ##################################')

#% Scraping Functions
def memory_usage(df):
    mu = df.memory_usage(deep=True)
    mu = mu.append(pd.Series(mu.sum(),['TOTAL']))
    return round(mu / 1024 ** 2, 2)

change_slashes = lambda path: print(path)

# def pickle_to_csv(filename):
#     t0=time.time()
#     csv_file = ''.join(filename.split('.')[:-1]) + '.csv'
#     df = read_pickle(DirInfo.TICK.path + filename, True)
#     print('Loaded the pickle, saving CSV...')
#     df.to_csv(DirInfo.TICK.path + csv_file, index=False)

#     print(f'Converted the {csv_file} file! Time took: {round(time.time()-t0,2)} secs')
#     return csv_file

def current_ms(): return int(time.time() * 1000.0)


def current_datetime(): return ms_to_datetime(current_ms())


# def current_milli_time(): return int(round(time.time() * 1000))


def interval_to_min(interval): return interval[:-1]


def datetime_to_ms(dt): return int(dt.timestamp() * 1000.0)


def ms_to_datetime(ms):
    return datetime.fromtimestamp(ms/1000.0)

# Returns none if doesn't exists!
# If not absolute, path relative to the DATA FOLDER (DirInfo.DATA.path)
def read_pickle(file_name, absolute_path=False):
    if not absolute_path: file_name = f'{DirInfo.DATA.path}{file_name}.pkl'
    pkl = None
    try:
        with open(file_name, 'rb') as f:
            pkl = pickle.load(f)
    except IOError:
        print(f"The file {file_name} does not exist!")
    finally:
        return pkl

# If not absolute, path relative to the DATA FOLDER (DirInfo.DATA.path)
def write_pickle(obj, file_name, absolute_path=False):
    if not absolute_path:
        file_name = f'{DirInfo.DATA.path}{file_name}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def ar_process(phi, nsample, sd = 1, init_val = 0):
    data = np.array([init_val] + [None]*(nsample-1), dtype=np.float32)
    for i in range(1,nsample):
        data[i] = data[i-1] * phi +  np.random.normal(0, sd)
    return data

def is_good_data(df, message=False):
    assert(not df.empty)
    # Check duplicates
    is_good = len(set(df.index)) == len(df.index)
    if is_good and message:
        print(f'No duplicates in {len(df.index)} rows of data!')
    else:
        print(f'Found {len(df.index)-len(set(df.index))} duplicate(s) out of {len(df.index)}!')
        dups = df[df.index.duplicated()].index
        for i in dups:
            loc = list(df.index).index(i)
            print(
                f'Found duplicate at {loc}/{len(df.index)}. {len(df.index)-loc} from the end.')

    # Check data gaps
    # df['']
    return is_good

# pickle_to_csv(DirInfo.TICK.path + 'BTCUSDT_TICK_20200101_20200827.pkl')

'''######################## Preprocessing Functions ######################## '''
def clean_data(df):
    # df = df.drop(df.select_dtypes(include=['datetime64']).columns, 1)
    # df = df.drop(df.columns[df.count() < int(len(df)*0.98)], axis=1)
    assert not np.any(np.isnan(df).all())

    df = df.astype(np.float32)
    df.dropna(axis = 1, how='all',inplace=True)
    df.dropna(axis = 0, how='any',inplace = True)

    df = df.replace([np.inf, -np.inf], np.nan)
    # df.fillna(method='ffill',inplace = True)

    assert np.all(np.isfinite(df)) # Get inf columns: df.columns[np.isfinite(df).all() ==False]
    assert not np.any(np.isnan(df)) # df.columns[np.isnan(df).any()]
    return df.sort_index()

def split_data(features_signals, train_ratio):
    dev_mark = round((train_ratio)*len(features_signals))
    test_mark = round((train_ratio+(1-train_ratio)/2)*len(features_signals))

    if type(features_signals) == np.ndarray:
        train = features_signals[:dev_mark]
        dev = features_signals[dev_mark:test_mark]
        test = features_signals[test_mark:]
    else:
        train = features_signals.iloc[:dev_mark]
        dev = features_signals.iloc[dev_mark:test_mark]
        test = features_signals.iloc[test_mark:]
    # train.shape, dev.shape, test.shape
    return train, dev, test

def split_data_stateful_ol(data, batch_size, seq_len, train_ratio=0.95):
    extra = seq_len - 1
    N = len(data)
    max_mark = (N - 2 * batch_size)
    dev_mark = int(N * train_ratio // batch_size * batch_size) + extra
    while dev_mark > max_mark:
        dev_mark = dev_mark - batch_size

    diff = N - dev_mark + extra
    assert diff / batch_size >= 2

    ratio = int(diff // batch_size - 1)
    test_mark = dev_mark  + ratio * batch_size - extra

    # if type(data) == np.ndarray:
    train = data[:dev_mark]
    dev = data[(dev_mark-extra):(test_mark+extra)]
    test = data[(test_mark-extra):(test_mark+batch_size)]
    # else:
        # train = data.iloc[:dev_mark]
        # dev = data.iloc[(dev_mark-extra):(test_mark-extra)]
        # test = data.iloc[(test_mark-extra):]
    # train.shape, dev.shape, test.shape
    return train, dev, test

def plot_ind(train, dev, test, ind, only_dist=False):
    if not only_dist:
        plt.figure(figsize=(16, 5))
        plt.title(f'{ind} KDE Plot')
        plt.plot(np.arange(train.shape[0]), train[ind], label=f'Train {ind}')
        plt.plot(np.arange(train.shape[0], train.shape[0] + dev.shape[0]), dev[ind], label=f'Dev {ind}')
        plt.plot(np.arange(train.shape[0]+dev.shape[0], train.shape[0] + dev.shape[0]+test.shape[0]), test[ind], label=f'Test {ind}')
        plt.legend(loc='upper left')

    plt.figure(figsize=(16, 5))
    plt.title(f'{ind} KDE Plot')
    sns.kdeplot(train[ind], label=f'Train {ind}')
    sns.kdeplot(dev[ind], label=f'Dev {ind}')
    sns.kdeplot(test[ind], label=f'Test {ind}')

def plot_tf_history(r, save=True):
    # print(r.history.keys())
    #  "Metric"
    if len(r.history.keys())>4:
        metric, val_metric = [x for i,x in enumerate(r.history.keys()) if i in [1,4]]
    else:
        metric, val_metric = [x for x in r.history.keys() if x not in ['loss', 'val_loss']]
    # plt.figure(figsize=(10, 5))
    plt.plot(r.history[metric])
    plt.plot(r.history[val_metric])
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.grid(True)
    if save:
        plt.savefig(DirInfo.PLOTS.path + f'acc_plot_{int(time()*1000)}.png')
    plt.show()

    # "Loss"
    # plt.figure(figsize=(10, 5))
    plt.plot(r.history['loss'])
    plt.plot(r.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.grid(True)
    if save:
        plt.savefig(DirInfo.PLOTS.path + f'loss_plot_{int(time()*1000)}.png')
    plt.show()

def timeseries_to_sequences(data, targets, seq_len, bs = 1024):
    # data.shape, targets.shape
    L, D = data.shape
    N = L - seq_len + 1
    if len(targets.shape) == 3:
        _,n_class,n_max = targets.shape
    else:
        _,n_class = targets.shape

    data_gen = np.append(data,[np.nan]*D).reshape(-1,D)
    if len(targets.shape) == 2:
        targets_gen = np.append(np.empty(shape=(n_class)),targets).reshape(-1,n_class)
        targets = np.empty((N, n_class), dtype=np.float32)
    elif len(targets.shape) == 3:
        targets_gen = np.append(np.empty(shape=(n_class, n_max)),targets).reshape(-1, n_class, n_max)
        targets = np.empty((N, n_class, n_max), dtype=np.float32)

    gen = TimeseriesGenerator(data_gen, targets_gen, seq_len, batch_size=bs)
    data = np.empty((N,seq_len,D), dtype=np.float32)

    for i in range(len(gen)):
        data[i*bs:(i+1)*bs], targets[i*bs:(i+1)*bs] = gen[i]

    assert data.shape == (N, seq_len, D)
    return data, targets

def balance_xy(x,y):
    counts = np.unique(y, return_counts=True)
    train_min = min(counts[1])
    print(f'Lost data for downsampling to {train_min} candles: {max(counts[1])-train_min}')

    mask = np.hstack([np.random.choice(np.where(y == l)[0], train_min, replace=False) for l in counts[0]])
    return x[mask], y[mask]

def one_hot_to_ind(data):
    if len(data.shape)==3:
        res = np.empty(shape=(data.shape[:-1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                res[i,j] = np.argmax(data[i,j])
    elif len(data.shape)==2:
        res = np.empty(shape=(data.shape[:-1]))
        for i in range(data.shape[0]):
                res[i] = np.argmax(data[i])
    elif len(data.shape)==1:
        return np.argmax(data)
    else:
        raise ValueError
    return res

'''######################## Stateful Functions ######################## '''
def match_batch_size(X,Y, batch_size):
    assert len(X) == len(Y)
    to_drop = len(X) % batch_size
    return X[to_drop:], Y[to_drop:]

if __name__ == "__main__":
    print('ready.')


# %%

