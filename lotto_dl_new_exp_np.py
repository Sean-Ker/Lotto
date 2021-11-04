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


def shift(xs, n, axis=-1):
    def shift_1d(xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = np.nan
            e[n:] = xs[:-n]
        else:
            e[n:] = np.nan
            e[:n] = xs[-n:]
        return e

    # ra = (xs.ndim-1) - (axis % (xs.ndim-1))
    return np.apply_along_axis(lambda x: shift_1d(x, n), axis, xs)


# assert np.equal(shift(train, 1, 0), df_train.shift(1).values)


def make_data_bits(df):
    data = df.values.copy()
    data = data.astype(np.uint8)
    data = np.unpackbits(data, axis=-1).astype(np.float32)
    return data


# constants
game = "dgrad"
NUM_LIM = (1, 49)
DGRAD_LIM = (1, 7)
TESTING_IND = 0.9
SEQ_LEN = 15

df = pd.read_csv("data/daily_grand/DailyGrand.csv", index_col=0)
df.reset_index(drop=True, inplace=True)
df.drop(["DRAW NUMBER", "DRAW DATE", "PRIZE DIVISION", "SEQUENCE NUMBER"], axis=1, inplace=True)
df.columns = ["n1", "n2", "n3", "n4", "n5", "ngrand"]
df_no_dgrand = df.drop("ngrand", axis=1)
df = df.astype(np.float32)
assert not df_no_dgrand.duplicated().any()

n_train = int(len(df) * TESTING_IND)

#%%
all_data = []

train = df.copy().iloc[:n_train]
test = df.copy().iloc[n_train:]
assert len(train) + len(test) == len(df)

num_columns = [x for x in train.columns if x != "ngrand"]
ngrand_col = [x for x in train if x not in num_columns]

y_train_god = train.copy()
y_test_god = test.copy()

y_train_god[num_columns] = (y_train_god[num_columns] - NUM_LIM[0]) / (NUM_LIM[1] - NUM_LIM[0])
y_train_god[ngrand_col] = y_train_god[ngrand_col].where(
    (y_train_god[ngrand_col] == 0), (y_train_god[ngrand_col] - DGRAD_LIM[0]) / (DGRAD_LIM[1] - DGRAD_LIM[0])
)

y_test_god[num_columns] = (y_test_god[num_columns] - NUM_LIM[0]) / (NUM_LIM[1] - NUM_LIM[0])
y_test_god[ngrand_col] = y_test_god[ngrand_col].where(
    (y_test_god[ngrand_col] == 0), (y_test_god[ngrand_col] - DGRAD_LIM[0]) / (DGRAD_LIM[1] - DGRAD_LIM[0])
)

y_train_god = y_train_god.values[1:]
y_test_god = y_test_god.values[1:]

for i in range(5):
    df_train = train.copy()
    df_test = test.copy()

    if i:
        for j in range(len(df_train)):
            df_train.iloc[j][num_columns] = np.random.choice(df_train[num_columns].iloc[j], 5, False)
        for j in range(len(df_test)):
            df_test.iloc[j][num_columns] = np.random.choice(df_test[num_columns].iloc[j], 5, False)
        print("Shuffled data!")

    train_bits = shift(make_data_bits(df_train), 1, 0)[1:].astype(np.uint8)
    test_bits = shift(make_data_bits(df_test), 1, 0)[1:].astype(np.uint8)

    df_train[num_columns] = (df_train[num_columns] - NUM_LIM[0]) / (NUM_LIM[1] - NUM_LIM[0])
    df_train[ngrand_col] = df_train[ngrand_col].where(
        (df_train[ngrand_col] == 0), (df_train[ngrand_col] - DGRAD_LIM[0]) / (DGRAD_LIM[1] - DGRAD_LIM[0])
    )

    df_test[num_columns] = (df_test[num_columns] - NUM_LIM[0]) / (NUM_LIM[1] - NUM_LIM[0])
    df_test[ngrand_col] = df_test[ngrand_col].where(
        (df_test[ngrand_col] == 0), (df_test[ngrand_col] - DGRAD_LIM[0]) / (DGRAD_LIM[1] - DGRAD_LIM[0])
    )

    # df_train.describe()

    #%%
    x_train = shift(df_train.copy().values, 1, 0)[1:]
    x_test = shift(df_test.copy().values, 1, 0)[1:]
    # x_train = x_train[~np.isnan(x_train)]

    x_train = np.concatenate([x_train, train_bits], axis=-1)
    x_test = np.concatenate([x_test, test_bits], -1)

    #%%
    x_train, y_train = util.timeseries_to_sequences(x_train, y_train_god.copy(), SEQ_LEN)
    x_test, y_test = util.timeseries_to_sequences(x_test, y_test_god.copy(), SEQ_LEN)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    all_data.append([x_train, y_train, x_test, y_test])


#%%
x_train, y_train, x_test, y_test = [np.concatenate(x, axis=0) for x in np.stack(all_data, -1)]
N, _, D = x_train.shape
n_target = y_train.shape[-1]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
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
    Flatten,
    concatenate,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


def build_model():

    inp = Input(shape=(SEQ_LEN, D))

    x = Dense(32, activation="relu")(inp)

    for _ in range(10):
        x = GRU(32, return_sequences=True)(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
    x = GRU(32)(x)

    # # densed = tf.transpose(inp, (0, 2, 1))
    # densed = Dense(32, activation='relu')(inp)
    # densed = Dense(16,activation='relu')(densed)
    # # densed = tf.transpose(x, (0, 2, 1))

    # print(densed.shape)
    # densed = Flatten()(densed)
    # print(densed.shape)

    # densed = Dropout(0.3)(densed)
    # densed = Dense(128,activation='relu')(densed)
    # densed = Dense(128,activation='relu')(densed)
    # densed = Dense(64,activation='relu')(densed)

    # x = concatenate([x, densed])
    # # x = Dropout(0.3)(x)
    # x = Dense(128,activation='relu')(x)
    # x = Dense(128,activation='relu')(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)

    x = Dense(16, activation="relu")(x)
    x = Dense(n_target)(x)

    return Model(inp, x)


model = build_model()
plot_model(model, show_shapes=True)
model.summary()
# %%
# Create a callback that saves the model's weights
checkpoint_path = f"saved_models\\saved_model.seqlen{SEQ_LEN}-" + "{val_loss:.4f}" + f"_{int(time.time())}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0, save_weights_only=False
)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3 / 200)
model.compile(loss="mse", optimizer=opt)

# tensorboard --logs=logs/fit
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

#%%
r = model.fit(
    x_train,
    y_train,
    batch_size=int((N / 2) + 1),
    verbose=1,
    callbacks=[cp_callback, tensorboard_callback],
    initial_epoch=0,
    epochs=1000,
    validation_data=(x_test, y_test),
    shuffle=True,
)

# %%
model = load_model("saved_models\\saved_model.seqlen15-0.0476_1612127789.h5")

y_pred = model.predict(x_test)
model.evaluate(x_test, y_test)

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
print(y_pred_t[:5], "\n\n", y_true[:5])
# y_true[:5]

# %%


# %%

