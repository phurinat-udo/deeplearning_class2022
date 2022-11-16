# %%
import numpy as np
num_timesteps = 100
num_inputs = 32 # number of features
num_outputs = 5 # number of targets
# %%
inputs = np.random.random((num_timesteps, num_inputs)) 
s_t = np.zeros((num_outputs,))
Wx = np.random.random((num_outputs, num_inputs))
Ws = np.random.random((num_outputs, num_outputs))
b = np.random.random((num_outputs,)) 
# %%
successive_outputs = []
for x_t in inputs:
    y_t = np.tanh(np.dot(Wx, x_t) + np.dot(Ws, s_t) + b)
    successive_outputs.append(y_t)
    s_t = y_t
final_output_sequence = np.stack(successive_outputs, axis=0)
# %%
import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN # convert data vector into matrix
def convertToMatrix(data, step):
    X, Y =[], []
    for i in range(len(data)-step):
        d=i+step; X.append(data[i:d,]); Y.append(data[d,])
    return np.array(X), np.array(Y)
step = 4; N = 1000; Tp = 800; t=np.arange(0,N) 
x=np.sin(0.02*t)+2*np.random.rand(N)
df = pd.DataFrame(x); df.head()
plt.plot(df); plt.show()
values=df.values;
train,test = values[0:Tp,:], values[Tp:N,:]
# add step elements into train and test
test = np.append(test,np.repeat(test[-1,],step)) 
train = np.append(train,np.repeat(train[-1,],step))
# %%
trainX,trainY = convertToMatrix(train,step)
testX,testY = convertToMatrix(test,step)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1])) 
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(1,step), activation="relu")) 
model.add(Dense(8, activation="relu")); model.add(Dense(1)) 
model.compile(loss='mean_squared_error', optimizer='rmsprop') 
model.summary()
# %%
model.fit(trainX,trainY, epochs=100, batch_size=16, verbose=2) 
trainPredict = model.predict(trainX)
testPredict= model.predict(testX) 
predicted=np.concatenate((trainPredict,testPredict),axis=0)
trainScore = model.evaluate(trainX, trainY, verbose=0) 
# %%
print(trainScore); index = df.index.values 
plt.plot(index,df); plt.plot(index,predicted) 
plt.axvline(df.index[Tp], c="r"); plt.show()
# %%
import os
fname = os.path.join("lect_originalEX/jena_climate_2009_2016.csv")
with open(fname) as f:
    data = f.read()
lines = data.split("\n") 
header = lines[0].split(",") 
lines = lines[1:] 
print(header) 
print(len(lines))
# %%
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1)) 
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]] 
    temperature[i] = values[1]
    raw_data[i, :] = values[:]
from matplotlib import pyplot as plt
plt.plot(range(len(temperature)), temperature)
# %%
plt.plot(range(1440), temperature[:1440])
# %%
n = len(raw_data)
num_train_samples = int(0.5 * n)
num_val_samples = int(0.25 * n)
num_test_samples = n - num_train_samples - num_val_samples 
print(num_train_samples, num_val_samples, num_test_samples)
# %%
mean = raw_data[:num_train_samples].mean(axis=0) 
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
# %%
from keras.utils import timeseries_dataset_from_array 
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256
train_dataset = timeseries_dataset_from_array(
raw_data[:-delay], targets=temperature[delay:], sampling_rate = sampling_rate, sequence_length=sequence_length, shuffle=True, batch_size=batch_size, start_index=0, end_index=num_train_samples)
# %%
val_dataset = timeseries_dataset_from_array( raw_data[:-delay], targets=temperature[delay:], sampling_rate=sampling_rate, sequence_length=sequence_length, shuffle=True,
batch_size=batch_size, start_index=num_train_samples, end_index=num_train_samples + num_val_samples)
test_dataset = timeseries_dataset_from_array( raw_data[:-delay], targets=temperature[delay:], sampling_rate=sampling_rate, sequence_length=sequence_length, shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)
# %%
# Checking the shape of data
for samples, targets in train_dataset: 
    print("samples shape:", samples.shape) 
    print("targets shape:", targets.shape) 
    break
# Create and train a model
from keras import Model
from keras.layers import Input, SimpleRNN, Dense
from keras.models import load_model
inputs = Input(shape=(sequence_length, raw_data.shape[-1]))
x = SimpleRNN(16)(inputs)
outputs = Dense(1)(x)
model = Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
h = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
# %%
plt.subplot(121) 
plt.plot(h.epoch,h.history["loss"]) 
plt.plot(h.epoch,h.history["val_loss"]) 
plt.title("MSE")

plt.subplot(122) 
plt.plot(h.epoch,h.history["mae"]) 
plt.plot(h.epoch,h.history["val_mae"]) 
plt.title("MAE")
# %%
