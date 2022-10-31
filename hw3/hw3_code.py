# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras import Sequential, callbacks
from keras import optimizers, models
from keras.layers import Dense, Flatten, Dropout, Input, Resizing
from keras.metrics import SparseCategoricalAccuracy
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# %%
data = tfds.load('citrus_leaves', split='train', batch_size=-1, as_supervised=True,)
x_train, y_train = tfds.as_numpy(data)
print(x_train.shape, y_train.shape, y_train[0:10])
num_classes = 4
x_train_cate = x_train.astype('float32')
y_train_cate = to_categorical(y_train, num_classes)
print(x_train.shape, y_train.shape, y_train[0:10])
print(x_train.dtype, y_train.dtype, y_train[0])
# %%
adam = optimizers.Adam(epsilon=0.00015, learning_rate=0.001, amsgrad=True, beta_1=0.8, beta_2=0.998)
# cb = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.05)
# %%
model_cate = Sequential()
model_cate.add(Input(shape=(256,256,3,)))
model_cate.add(Flatten())
model_cate.add(Dense(200, activation='relu'))
# model_cate.add(Dense(200, activation='relu'))
# model_cate.add(Dense(200, activation='relu'))
model_cate.add(Dense(100, activation='relu'))
# model_cate.add(Dense(100, activation='relu'))
model_cate.add(Dense(4, activation="softmax"))
model_cate.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["acc"])
model_cate.summary()
# %%
model_sparcat = Sequential()
model_sparcat.add(Input(shape=(256,256,3,)))
model_sparcat.add(Flatten())
model_sparcat.add(Dense(200, activation='relu'))
# model_sparcat.add(Dense(200, activation='relu'))
# model_sparcat.add(Dense(200, activation='relu'))
model_sparcat.add(Dense(100, activation='relu'))
# model_sparcat.add(Dense(100, activation='relu'))
model_sparcat.add(Dense(4, activation="softmax"))
model_sparcat.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=[SparseCategoricalAccuracy(name="acc")])
model_sparcat.summary()
# %%
t1 = time.time()
h_cate = model_cate.fit(x_train_cate, y_train_cate, batch_size=40, epochs=200, validation_split=0.2)
h_sparcat = model_sparcat.fit(x_train, y_train, batch_size=40, epochs=200, validation_split=0.2)
t2 = time.time()
print(f"Runtime = {t2-t1} s")
# %%
fig_mlp = plt.figure(figsize=(13,10), tight_layout=True)
ax_mlpl = fig_mlp.add_subplot(221)
# ax_mlpl.plot(h_cate.epoch, h_cate.history['val_loss'], '-', label='Valid Categorical')
# ax_mlpl.plot(h_cate.epoch, h_cate.history['loss'], '-', label='Train Categorical')
ax_mlpl.plot(h_sparcat.epoch, h_sparcat.history['loss'], '-', label='Train Sparse Categorical')
ax_mlpl.plot(h_sparcat.epoch, h_sparcat.history['val_loss'], '-', label='Valid Sparse Categorical')
ax_mlpl.set_title('Cross Entopy Loss function')
ax_mlpl.set_xlabel('Iteration number')
ax_mlpl.set_ylabel('loss')
import pandas as pd
acc_mlp = pd.DataFrame(h_sparcat.history['val_acc'], h_sparcat.epoch)
ax_mlp = fig_mlp.add_subplot(222)
# ax_mlp.plot(h_cate.epoch, h_cate.history['acc'], label='Train Categorical')
# ax_mlp.plot(h_cate.epoch, h_cate.history['val_acc'], label='Valid Categorical')
ax_mlp.plot(h_sparcat.epoch, h_sparcat.history['acc'], label='Train Sparse Categorical')
ax_mlp.plot(acc_mlp.index, acc_mlp.values, label='Valid Sparse Categorical')
for i, y in acc_mlp.iterrows():
    if y.values > 0.8:
        ax_mlp.annotate(f'{y.values}', xy=(i,y.values), textcoords='data')
ax_mlp.set_title('Accuracy')
ax_mlp.set_xlabel('Iteration number')
ax_mlp.set_ylabel('Accuracy')
plt.legend()
fig_mlp.suptitle('MLP with Adam Method')

fig_mlp.savefig('hw3_prob1_21grad_b108_labeled.png', dpi=300, bbox_inches='tight')
# %%
# model_cate.save('model_catexentropy_21_grad_b108_2')
# model_sparcat.save('model_spars_catexentropy_21_grad_b108_2')

# %%
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Rescaling, RandomFlip, RandomRotation, RandomZoom
adam_cnn = optimizers.Adam(epsilon=0.0001, learning_rate=0.001)
model_CNN = Sequential()
model_CNN.add(Input(shape=(256,256,3)))
model_CNN.add(Rescaling(scale=1.0/255))
model_CNN.add(RandomFlip("horizontal"))
model_CNN.add(RandomRotation([-0.5,0.5]))
model_CNN.add(RandomZoom([-0.5,0]))
model_CNN.add(Conv2D(16, kernel_size=(3, 3), activation="relu"))
model_CNN.add(AveragePooling2D(pool_size=(2, 2)))
# model_CNN.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
# model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
# model_CNN.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
# model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Flatten())
model_CNN.add(Dense(50, activation="relu"))
model_CNN.add(Dense(4, activation="softmax"))
# model_CNN.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=[SparseCategoricalAccuracy(name="acc")])
model_CNN.compile(optimizer=adam_cnn, loss="categorical_crossentropy", metrics=["acc"])
model_CNN.summary()
# %%
cb_cnn = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.1)
# %%
t1 = time.time()
# h_CNN = model_CNN.fit(x_train, y_train, batch_size=40, epochs=200, validation_split=0.2, callbacks=cb_cnn)
h_CNN = model_CNN.fit(x_train_cate, y_train_cate, batch_size=40, epochs=200, validation_split=0.2, callbacks=cb_cnn)
t2 = time.time()
print(f"Runtime = {t2-t1} s")
# %%
fig_cnn = plt.figure(figsize=(13,10), tight_layout=True)
ax_cnnl = fig_cnn.add_subplot(221)
ax_cnnl.plot(h_CNN.epoch, h_CNN.history['loss'], '-', label='Train Categorical')
ax_cnnl.plot(h_CNN.epoch, h_CNN.history['val_loss'], '-', label='Valid Categorical')
ax_cnnl.set_title('Categorical Crossentopy Loss function')
ax_cnnl.set_xlabel('Iteration number')
ax_cnnl.set_ylabel('loss')

ax_cnn = fig_cnn.add_subplot(222)
ax_cnn.plot(h_CNN.epoch, h_CNN.history['acc'], label='Train Categorical')
ax_cnn.plot(h_CNN.epoch, h_CNN.history['val_acc'], label='Valid Categorical')
ax_cnn.set_title('Accuracy')
ax_cnn.set_xlabel('Iteration number')
ax_cnn.set_ylabel('Accuracy')
plt.legend()
fig_cnn.suptitle('CNN with Adam Method')
fig_cnn.savefig('hw3_prob2_Augmented.png', dpi=300, bbox_inches='tight')
# %%
model_CNN.save('CNN_model')
# %%
model_MLP_cat = Sequential()
model_MLP_cat.add(Input(shape=(256,256,3,)))
# model_MLP_cat.add(Rescaling(scale=1.0/255))
model_MLP_cat.add(Flatten())
model_MLP_cat.add(Dense(300, activation='relu'))
# model_MLP_cat.add(Dense(200, activation='relu'))
# model_MLP_cat.add(Dense(200, activation='relu'))
# model_MLP_cat.add(Dense(100, activation='relu'))
model_MLP_cat.add(Dense(100, activation='relu'))
model_MLP_cat.add(Dropout(0.02))
model_MLP_cat.add(Dense(4, activation="softmax"))
model_MLP_cat.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["acc"])
model_MLP_cat.summary()
# %%
t1 = time.time()
h_MLP_cat = model_MLP_cat.fit(x_train_cate, y_train_cate, batch_size=40, epochs=200, validation_split=0.2, callbacks=cb_cnn)
t2 = time.time()
print(f"Runtime = {t2-t1} s")
# %%
fig_mlp = plt.figure(figsize=(13,10), tight_layout=True)
ax_mlpl = fig_mlp.add_subplot(221)
ax_mlpl.plot(h_MLP_cat.epoch, h_MLP_cat.history['val_loss'], '-', label='Valid Categorical')
ax_mlpl.plot(h_MLP_cat.epoch, h_MLP_cat.history['loss'], '-', label='Train Categorical')
ax_mlpl.set_title('Cross Entopy Loss function')
ax_mlpl.set_xlabel('Iteration number')
ax_mlpl.set_ylabel('loss')

ax_mlp = fig_mlp.add_subplot(222)
ax_mlp.plot(h_MLP_cat.epoch, h_MLP_cat.history['acc'], label='Train Categorical')
ax_mlp.plot(h_MLP_cat.epoch, h_MLP_cat.history['val_acc'], label='Valid Categorical')
ax_mlp.set_title('Accuracy')
ax_mlp.set_xlabel('Iteration number')
ax_mlp.set_ylabel('Accuracy')
plt.legend()
fig_mlp.suptitle('MLP with Adam Method')