# %%
import pickle
import numpy as np
from keras import Sequential, layers
from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
# %%
X = np.linspace(0, 1, 20)
y = np.exp(-X) * np.cos(2*np.pi*X)
print(f'{[i for i in X]}-{[i for i in y]}')
# %%
fig_data = plt.figure(figsize=(2,1))
ax_dat = fig_data.add_axes([0.01, 0.01, 3, 3])
ax_dat.plot(X, y, '-o')
ax_dat.set_title('Data points')
ax_dat.set_xlabel('X')
ax_dat.set_ylabel('y')
fig_data.savefig('hw2_data1.png', dpi=300, bbox_inches='tight')
# %%
model = Sequential()
sgd = optimizers.SGD(learning_rate=0.01)
model.add(layers.Input(shape=(1,)))
model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dense(1))
model.compile(loss='mse', optimizer=sgd)
t1 = time.time()
hist_bgd = model.fit(X, y, epochs=2000, verbose=0, batch_size=20); t2 = time.time()
print(f"Runtime = {t2-t1} s")
# %%
fig_bgd = plt.figure(figsize=(13,10), tight_layout=True)
ax_bgd = fig_bgd.add_subplot(221)
ax_bgd.plot(X, y, '-o')
ax_bgd.plot(X, hist_bgd.model(X), '-')
ax_bgd.set_title('Data points and fit')
ax_bgd.set_xlabel('X')
ax_bgd.set_ylabel('y')

ax_bgdl = fig_bgd.add_subplot(222)
ax_bgdl.plot(hist_bgd.epoch, hist_bgd.history['loss'])
ax_bgdl.set_title('MSE Loss Function')
ax_bgdl.set_xlabel('Iteration number')
ax_bgdl.set_ylabel('loss')
fig_bgd.suptitle('MLP with Batch Gradient Descent Method')
fig_bgd.savefig('hw2_prob1.png', dpi=300, bbox_inches='tight')
# %%
model2 = Sequential()
sgd = optimizers.SGD(learning_rate=0.01)
model2.add(layers.Input(shape=(1,)))
model2.add(layers.Dense(2, activation='sigmoid'))
model2.add(layers.Dense(1))
model2.compile(loss='mse', optimizer=sgd)
t1 = time.time()
hist_sgd = model2.fit(X, y, epochs=3000, verbose=0, batch_size=1); t2 = time.time()
print(f"Runtime = {t2-t1} s")
# %%
fig_sgd = plt.figure(figsize=(13,10), tight_layout=True)
ax_sgd = fig_sgd.add_subplot(221)
ax_sgd.plot(X, y, '-o')
ax_sgd.plot(X, hist_sgd.model(X), '-')
ax_sgd.set_title('Data points and fit')
ax_sgd.set_xlabel('X')
ax_sgd.set_ylabel('y')

ax_sgdl = fig_sgd.add_subplot(222)
ax_sgdl.plot(hist_sgd.epoch, hist_sgd.history['loss'])
ax_sgdl.set_title('MSE Loss Function')
ax_sgdl.set_xlabel('Iteration number')
ax_sgdl.set_ylabel('loss')
fig_sgd.suptitle('MLP with Stochastic Gradient Descent Method')
fig_sgd.savefig('hw2_prob2.png', dpi=300, bbox_inches='tight')
# %%
model3 = Sequential()
sgd = optimizers.SGD(learning_rate=0.01)
model3.add(layers.Input(shape=(1,)))
model3.add(layers.Dense(1, activation='sigmoid'))
model3.add(layers.Dense(1, activation='sigmoid'))
model3.add(layers.Dense(1))
model3.compile(loss='mse', optimizer=sgd)
t1 = time.time()
hist_mgd = model3.fit(X, y, epochs=1000, verbose=0, batch_size=5); t2 = time.time()
print(f"Runtime = {t2-t1} s")
# %%
fig_mgd = plt.figure(figsize=(13,10), tight_layout=True)
ax_mgd = fig_mgd.add_subplot(221)
ax_mgd.plot(X, y, '-o')
ax_mgd.plot(X, hist_mgd.model(X), '-')
ax_mgd.set_title('Data points and fit')
ax_mgd.set_xlabel('X')
ax_mgd.set_ylabel('y')

ax_mgdl = fig_mgd.add_subplot(222)
ax_mgdl.plot(hist_mgd.epoch, hist_mgd.history['loss'])
ax_mgdl.set_title('MSE Loss Function')
ax_mgdl.set_xlabel('Iteration number')
ax_mgdl.set_ylabel('loss')
fig_mgd.suptitle('MLP with Mini-batch Gradient Descent Method')
fig_mgd.savefig('hw2_prob3.png', dpi=300, bbox_inches='tight')
# %%
model4 = Sequential()
sgd = optimizers.SGD(learning_rate=0.05)
model4.add(layers.Input(shape=(1,)))
model4.add(layers.Dense(1, activation='relu'))
model4.add(layers.Dense(2, activation='sigmoid'))
model4.add(layers.Dense(2, activation='relu'))
model4.add(layers.Dense(3, activation='sigmoid'))
model4.add(layers.Dense(1))
model4.compile(loss='mse', optimizer=sgd)
t1 = time.time()
hist_bon = model4.fit(X, y, epochs=2000, verbose=0, batch_size=2); t2 = time.time()
print(f"Runtime = {t2-t1} s")
# %%
fig_bon = plt.figure(figsize=(13,10), tight_layout=True)
ax_bon = fig_bon.add_subplot(221)
ax_bon.plot(X, y, '-o')
ax_bon.plot(X, hist_bon.model(X), '-')
ax_bon.set_title('Data points and fit')
ax_bon.set_xlabel('X')
ax_bon.set_ylabel('y')

ax_bonl = fig_bon.add_subplot(222)
ax_bonl.plot(hist_bon.epoch, hist_bon.history['loss'])
ax_bonl.set_title('MSE Loss Function')
ax_bonl.set_xlabel('Iteration number')
ax_bonl.set_ylabel('loss')
fig_bon.suptitle('MLP with Mini-batch Gradient Descent Method')
fig_bon.savefig('hw2_bonus.png', dpi=300, bbox_inches='tight')
# %%
# model4.save('bestfit_extramodel')
# %%
from keras import models
model5 = models.load_model('bestfit_extramodel')
plt.plot(X, model5(X))
# %%
from ann_visualizer.visualize import ann_viz;
import graphviz
g = graphviz.Graph(format='png')  
ann_viz(model5, view=True, filename='network.gv', title="Extra Credit Model")
# %%
with open('/', 'wb') as model_bon_best:
    pickle.dump(hist_bon.history, model_bon_best)
# %%
np.save('history1.npy',hist_bon.history)
# %%
type(hist_bon)
# %%
from keras import callbacks
filename='best_fit_history.csv'
history_logger = callbacks.CSVLogger(filename, separator=",", append=True)

# %%
history_logger.append(hist_bon)
# %%
hist_best = np.load('history1.npy',allow_pickle='TRUE').item()
type(hist_best)
# %%
print(hist_best)
# %%
hist_best.keys()
# %%
