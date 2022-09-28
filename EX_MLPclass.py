# %% Import library
from unicodedata import name
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from keras import Sequential, layers
from keras import optimizers
from sklearn.model_selection import train_test_split
# %%
raw = pd.read_csv('AqSolDB.csv')
raw
# %%
rawnp = raw.to_numpy()
rawnp
# %%
X = raw.iloc[:, 9:].astype(np.float32)
y = raw.iloc[:, 5]
px.line(y).show()
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.15)
# %%
model = Sequential()
model.add(layers.Input(shape=17,))
model.add(layers.Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
history = model.fit(X_train, y_train, validation_split=0.15, epochs=100, verbose=0)
# %%
from plotly.subplots import make_subplots
fig_nonFeatScal = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Mean Absolute Error"))
fig_nonFeatScal.add_trace(go.Scatter(x = history.epoch, y = history.history['loss'], name='loss'), row=1, col=1)
fig_nonFeatScal.add_trace(go.Scatter(x = history.epoch, y = history.history['val_loss'], name = 'val_loss'), row=1, col=1)
fig_nonFeatScal.add_trace(go.Scatter(x = history.epoch, y = history.history['mean_absolute_error'], name = 'MAE'), row=1, col=2) 
fig_nonFeatScal.add_trace(go.Scatter(x = history.epoch, y = history.history['val_mean_absolute_error'], name='VAL_MAE'), row=1, col=2) 
fig_nonFeatScal.show()
# %%
model.save('model_mlp1')
# %%
X_n = pd.DataFrame()
X_s = pd.DataFrame()
for i, idx in enumerate(X.columns):
    n = {}
    s = {}
    x_min = min(X.loc[:, idx])
    x_max = max(X.loc[:, idx])
    mean = np.mean(X.loc[:, idx])
    SD = np.std(X.loc[:, idx])
    for j, jdx in enumerate(X.loc[:, idx]):
        n[j] = ((X.loc[j, idx] - x_min))/(x_max - x_min)
        s[j] = (X.loc[j, idx] - mean)/SD
    X_n[idx] = n.values()
    X_s[idx] = s.values()
# %%
Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_n, y, random_state=0, test_size=0.15)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_s, y, random_state=0, test_size=0.15)
# %%
model2 = Sequential()
model2.add(layers.Input(shape=17,))
model2.add(layers.Dense(1))
model2.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
history2 = model2.fit(Xn_train, yn_train, validation_split=0.15, epochs=100, verbose=0)
# %%
model3 = Sequential()
model3.add(layers.Input(shape=17,))
model3.add(layers.Dense(1))
model3.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
history3 = model3.fit(Xs_train, ys_train, validation_split=0.15, epochs=100, verbose=0)
# %%
fig_FeatScal = make_subplots(rows=3, cols=2, subplot_titles=("Loss", "Mean Absolute Error"))
fig_FeatScal.add_trace(go.Scatter(x = history.epoch, y = history.history['loss'], name = 'loss'), row=1, col=1)
fig_FeatScal.add_trace(go.Scatter(x = history.epoch, y = history.history['val_loss'], name = 'val_loss'), row=1, col=1)
fig_FeatScal.add_trace(go.Scatter(x = history.epoch, y = history.history['mean_absolute_error'], name = 'MAE'), row=1, col=2) 
fig_FeatScal.add_trace(go.Scatter(x = history.epoch, y = history.history['val_mean_absolute_error'], name='VAL_MAE'), row=1, col=2) 
fig_FeatScal.add_trace(go.Scatter(x = history2.epoch, y = history2.history['loss'], name='loss'), row=2, col=1)
fig_FeatScal.add_trace(go.Scatter(x = history2.epoch, y = history2.history['val_loss'], name = 'val_loss'), row=2, col=1)
fig_FeatScal.add_trace(go.Scatter(x = history2.epoch, y = history2.history['mean_absolute_error'], name = 'MAE'), row=2, col=2) 
fig_FeatScal.add_trace(go.Scatter(x = history2.epoch, y = history2.history['val_mean_absolute_error'], name='VAL_MAE'), row=2, col=2) 
fig_FeatScal.add_trace(go.Scatter(x = history3.epoch, y = history3.history['loss'], name='loss'), row=3, col=1)
fig_FeatScal.add_trace(go.Scatter(x = history3.epoch, y = history3.history['val_loss'], name = 'val_loss'), row=3, col=1)
fig_FeatScal.add_trace(go.Scatter(x = history3.epoch, y = history3.history['mean_absolute_error'], name = 'MAE'), row=3, col=2) 
fig_FeatScal.add_trace(go.Scatter(x = history3.epoch, y = history3.history['val_mean_absolute_error'], name='VAL_MAE'), row=3, col=2) 
fig_FeatScal.show()
# %%
import plotly.io as pio
pio.write_html(fig_FeatScal, "feat_scaling.html")

# %%
for a, adx in enumerate(X_n.columns):
    print(np.mean(X_s.loc[:, adx]), end='___')
    print(np.std(X_s.loc[:, adx]))
    print(max(X_n.loc[:, adx]), end='___')
    print(min(X_n.loc[:, adx]))
# %%
