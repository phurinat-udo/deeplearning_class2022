# %%
import numpy as np
from keras import Sequential, layers
from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %%
X = np.linspace(0, 1, 30)
y = np.sin(2* np.pi * X)*np.exp(-2*X)

fig_data = plt.figure(figsize=(15, 10))
ax = fig_data.add_axes([0.01, 0.01, 0.5, 0.5])
ax.plot(X, y)
fig_data.show()
# %%
model = Sequential()
model.add(layers.Input(shape=(1,)))
model.add(layers.Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit(X, y, epochs=1000, verbose=0, batch_size=1)
# %% Plot loss
fig_hist1 = plt.figure()
ax1 = fig_hist1.add_axes([0.01, 0.01, 1, 1])
ax1.plot(history.epoch, history.history['loss'], label='SGD')
# ax1.plot(hBGD.epoch, hBGD.history['loss'], label='BGD')
# ax1.plot(hMGD.epoch, hMGD.history['loss'], label='MGD')
# ax1.plot(hCM.epoch, hCM.history['loss'], label='MGD_CM')
# ax1.plot(hNAG.epoch, hNAG.history['loss'], label='MGD_NAG')
fig_hist1.legend()
fig_hist1.show()
# %%
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# %%
yp = history.model(X)
fig_fit = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Fit"))
fig_fit.add_trace(go.Scatter(x = history.epoch, y = history.history['loss'], name='loss'), row=1, col=1)
fig_fit.add_trace(go.Scatter(x = X, y = y, name = 'data'), row=1, col=2)
fig_fit.add_trace(go.Scatter(x = X, y = yp, name = 'fit line'), row=1, col=2) 
fig_fit.show()
# %%
model2 = Sequential()
model2.add(layers.Input(shape=(1,)))
model2.add(layers.Dense(10, activation='sigmoid'))
model2.add(layers.Dense(10, activation='sigmoid'))
model2.add(layers.Dense(10, activation='sigmoid'))
model2.add(layers.Dense(10, activation='sigmoid'))
model2.add(layers.Dense(10, activation='sigmoid'))
model2.add(layers.Dense(10, activation='sigmoid'))
model2.add(layers.Dense(10, activation='sigmoid'))
model2.add(layers.Dense(1))
model2.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
history2 = model2.fit(X, y, epochs=1000, verbose=0, batch_size=5)
# %%
fig_fit2 = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Fit"))
fig_fit2.add_trace(go.Scatter(x = history2.epoch, y = history2.history['loss'], name='loss'), row=1, col=1)
fig_fit2.add_trace(go.Scatter(x = X, y = y, name = 'data'), row=1, col=2)
fig_fit2.add_trace(go.Scatter(x = X, y = history2.model(X), name = 'fit line'), row=1, col=2) 
fig_fit2.show()
# %%
plt.plot(X,y)
plt.plot(X, yp)
plt.plot(X, history2.model(X))

# %%
