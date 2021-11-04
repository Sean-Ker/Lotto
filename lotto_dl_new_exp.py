# %%
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

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

from utils import util
import os

# constants
game = "dgrad"
NUM_LIM = (1, 49)
DGRAD_LIM = (1, 7)
TESTING_IND = 0.9
SEQ_LEN = 30

df = pd.read_csv("data/daily_grand/DailyGrand.csv", index_col=0)
df.reset_index(drop=True, inplace=True)
df.drop(["DRAW NUMBER", "DRAW DATE", "PRIZE DIVISION", "SEQUENCE NUMBER"], axis=1, inplace=True)
df.columns = ["n1", "n2", "n3", "n4", "n5", "ngrand"]
df_no_dgrand = df.drop("ngrand", axis=1)
df = df.astype(np.float32)
assert not df_no_dgrand.duplicated().any()

n_train = int(len(df) * TESTING_IND)

df_train = df.iloc[:n_train]
df_test = df.iloc[n_train:]
assert len(df_train) + len(df_test) == len(df)

# def create_groups(df):
#     data = df.values.astype(np.int)
#     groups = []
#     all_nums = []
#     all_grands = []
#     for row in data:
#         groups.append(row[:-1])
#         all_nums += list(row[:-1])
#         if row[-1] != 0:
#             groups.append(int(row[-1]))
#             all_grands.append(int(row[-1]))
#     return groups, all_nums, all_grands

# groups_y, all_nums, all_grands = create_groups(df_train)
# groups_test, all_nums_test, all_grands_test = create_groups(df_test)

num_columns = [x for x in df_train.columns if x != "ngrand"]
ngrand_col = [x for x in df_train if x not in num_columns]

df_train[["promotion"]] = (df_train[ngrand_col] == 0).astype(int)
df_test[["promotion"]] = (df_test[ngrand_col] == 0).astype(int)

#%%
def make_data_bits(df):
    data = df.values.copy()
    data = data.astype(np.uint8)
    a = np.unpackbits(data, axis=-1)
    return a


# np.concatenate(df_train, )
train_bits = make_data_bits(df_train.iloc[:, :-1])
test_bits = make_data_bits(df_test.iloc[:, :-1])
#%%
df_train[num_columns] = (df_train[num_columns] - NUM_LIM[0]) / (NUM_LIM[1] - NUM_LIM[0])
df_train[ngrand_col] = df_train[ngrand_col].where(
    df_train["promotion"].astype(bool), (df_train[ngrand_col] - DGRAD_LIM[0]) / (DGRAD_LIM[1] - DGRAD_LIM[0])
)

df_test[num_columns] = (df_test[num_columns] - NUM_LIM[0]) / (NUM_LIM[1] - NUM_LIM[0])
df_test[ngrand_col] = df_test[ngrand_col].where(
    df_test["promotion"].astype(bool), (df_test[ngrand_col] - DGRAD_LIM[0]) / (DGRAD_LIM[1] - DGRAD_LIM[0])
)

# # df_train.describe()
# train = df_train.values
# test = df_test[].values

#%%

#%%
x_train, y_train = df_train.copy(), df_train.copy()
x_test, y_test = df_test.copy(), df_test.copy()

y_train.drop("promotion", axis=1, inplace=True)
y_test.drop("promotion", axis=1, inplace=True)

not_promotion = [c for c in x_train if c != "promotion"]
x_train[not_promotion] = x_train[not_promotion].shift(1)
x_test[not_promotion] = x_test[not_promotion].shift(1)

x_train.dropna(inplace=True)
x_test.dropna(inplace=True)

y_train = y_train.iloc[-len(x_train) :]
y_test = y_test.iloc[-len(x_test) :]

x_train, y_train = util.timeseries_to_sequences(x_train, y_train, SEQ_LEN)
x_test, y_test = util.timeseries_to_sequences(x_test, y_test, SEQ_LEN)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
N, _, D = x_train.shape
# print(x_train)s
# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    GlobalMaxPool1D,
    Input,
    BatchNormalization,
    Layer,
    MaxPool1D,
    MaxPool2D,
    TimeDistributed,
    Softmax,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


def build_model():

    inp = Input(shape=(SEQ_LEN, D))

    x = Dense(32)(inp)

    for _ in range(10):
        x = LSTM(32, return_sequences=True)(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    # print(x.shape)

    # x = tf.transpose(x, (0, 2, 1))
    # x = Dense(config.n_class, activation="relu")(x)
    # x = Dense(config.n_class)(x)
    # x = tf.transpose(x, (0, 2, 1))
    # x = Linear(config.n_class)(x)
    # x = Dense(64, activation='relu')(x)
    x = Dense(16)(x)
    x = Dense(D - 1)(x)
    # x = Dense(config.n_max, activation='softmax', use_bias=False)(x)
    # x = Dense(1, activation="relu")(x)

    return Model(inp, x)


model = build_model()
plot_model(model, show_shapes=True)
model.summary()
# %%
# Create a callback that saves the model's weights
checkpoint_path = f"saved_models\\saved_model.seqlen{SEQ_LEN}-" + "{val_loss:.4f}" + f"_{int(time.time())}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0, save_weights_only=False
)

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=1000, decay_rate=0.9, staircase=False)
# opt = tf.keras.optimizers.Adam(,decay=...)
opt = tf.keras.optimizers.Adam(learning_rate=8e-4, decay=1e-6)
model.compile(loss="MSE", optimizer=opt)

# tensorboard --logs=logs/fit
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

#%%
r = model.fit(
    x_train,
    y_train,
    batch_size=N,
    verbose=1,
    callbacks=[cp_callback, tensorboard_callback],
    initial_epoch=2000,
    epochs=4000,
    validation_data=(x_test, y_test),
    shuffle=True,
)

# %%
# model = load_model('saved_models\\saved_model.seqlen10-0.04_1611442898.hdf5')

y_pred = model.predict(x_test)
model.evaluate(x_test, y_test, use_multiprocessing=True)

# %%
def transform(data):
    data = data.copy()
    data[:, :5] = np.round(data[:, :5] * (NUM_LIM[1] - NUM_LIM[0]) + NUM_LIM[0])
    data[:, 5] = np.round(data[:, 5] * (DGRAD_LIM[1] - DGRAD_LIM[0]) + DGRAD_LIM[0])
    return data.astype(np.int)
    # (df_train[num_columns] - NUM_LIM[0]) / ( NUM_LIM[1] - NUM_LIM[0])
    # (df_train[ngrand_col] - DGRAD_LIM[0]) / (DGRAD_LIM[1] - DGRAD_LIM[0]))


y_pred_t = transform(y_pred)
y_true = transform(y_test)

y_pred_t.shape, y_true.shape
# y_pred_t[:5]
# y_true[:5]

# %%
