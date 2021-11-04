import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# The input and output, i.e. truth table, of a NAND gate
x_train = np.array([[0,0],[0,1],[1,0],[1,1]], "uint8")
y_train = np.array([[1],[1],[1],[0]], "uint8")

# Add layers to the model
inp = Input(shape=(2))
x=Dense(3, activation='relu')(inp)
x=Dense(3, activation='relu')(x)
x=Dense(1, activation='sigmoid')(x)

model = Model(inp, x)

# Compile the neural networks model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the neural networks model
model.fit(x_train, y_train, epochs=5000, verbose=1)
# Test the output of the trained neural networks based NAND gate
y_predict = model.predict(x_train)
print(y_predict)

model.save('saved_models\\nand_model.h5')
