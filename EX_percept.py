# %%
from keras import Sequential, layers
import numpy as np
from matplotlib import pyplot as plt
from keras import optimizers
# %%
n = 50
x = np.linspace(0, 1, n)
a = 5; b = -4; c = 3
y = a*x*x + b*x + c + 0.5*np.random.randn(n)

# %% MGD, BGD, SGD
model = Sequential()
model.add(layers.Dense(1, input_shape=(1, )))
sgd = optimizers.SGD(learning_rate=0.01)
model.summary()
model.compile(loss='mse', optimizer=sgd)
hMGD = model.fit(x, y, epochs=100, batch_size=10)
model = Sequential()
model.add(layers.Dense(1, input_shape=(1, )))
sgd = optimizers.SGD(learning_rate=0.01)
model.summary()
model.compile(loss='mse', optimizer=sgd)
hBGD = model.fit(x, y, epochs=100, batch_size=50)
model = Sequential()
model.add(layers.Dense(1, input_shape=(1, )))
sgd = optimizers.SGD(learning_rate=0.01)
model.summary()
model.compile(loss='mse', optimizer=sgd)
hSGD = model.fit(x, y, epochs=100, batch_size=1)

# %% Classical momentum
cm = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
model = Sequential()
model.add(layers.Dense(1, input_shape=(1, )))
model.summary()
model.compile(loss='mse', optimizer=cm)
hCM = model.fit(x, y, epochs=100, batch_size=10)

# %% NAG
nag = optimizers.SGD(learning_rate=0.01, momentum=0.8, nesterov=True)
model = Sequential()
model.add(layers.Dense(1, input_shape=(1, )))
model.summary()
model.compile(loss='mse', optimizer=nag)
hNAG = model.fit(x, y, epochs=100, batch_size=10)

# %% Build Learning rate scheduler
from keras.callbacks import LearningRateScheduler
def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return 0.9*lr
lr_scheduler = LearningRateScheduler(schedule)

# %%
# %% Plot loss
fig_sgd = plt.figure()
ax1 = fig_sgd.add_axes([0.01, 0.01, 1, 1])
ax1.plot(hSGD.epoch, hSGD.history['loss'], label='SGD')
ax1.plot(hBGD.epoch, hBGD.history['loss'], label='BGD')
ax1.plot(hMGD.epoch, hMGD.history['loss'], label='MGD')
ax1.plot(hCM.epoch, hCM.history['loss'], label='MGD_CM')
ax1.plot(hNAG.epoch, hNAG.history['loss'], label='MGD_NAG')
fig_sgd.legend()
fig_sgd.show()
# %%
x_plot = np.linspace(x.min(), x.max(), 1000)
y_sgd = hSGD.model(x_plot)
y_bgd = hBGD.model(x_plot)
y_mgd = hMGD.model(x_plot)
y_CM = hCM.model(x_plot)
y_NAG = hNAG.model(x_plot)
# %%
fig_fitline = plt.figure()
ax2 = fig_fitline.add_axes([0.01, 0.01, 1, 1])
ax2.scatter(x, y, label='Data point')
ax2.plot(x_plot, y_sgd, label='SGD')
ax2.plot(x_plot, y_bgd, label='BGD')
ax2.plot(x_plot, y_mgd, label='MGD')
ax2.plot(x_plot, y_CM, label='MGD_CM')
ax2.plot(x_plot, y_NAG, label='MGD_NAG')
fig_fitline.legend()
fig_fitline.show()
# %%
adagrad = optimizers.Adagrad()