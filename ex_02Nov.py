
# %%
import tensorflow as tf; import os; from matplotlib.pyplot import *
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip' 
path = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True) 
PATH = os.path.join(os.path.dirname(path), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
BATCH_SIZE = 32; IMG_SIZE = (160, 160)
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE) 
class_names = train_dataset.class_names
figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        subplot(3, 3, i + 1) 
        imshow(images[i].numpy().astype("uint8")) 
        title(class_names[labels[i]]); axis("off")
# %%
val_batches = tf.data.experimental.cardinality(validation_dataset) 
test_dataset = validation_dataset.take(val_batches // 5) 
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
# Number of validation batches: 26
# Number of test batches: 6
# %%
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
# %%
data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(0.2), ])
for image, _ in train_dataset.take(1): 
    figure(figsize=(10, 10)); first_image = image[0]
    for i in range(9):
        subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        imshow(augmented_image[0] / 255); axis('off')
# %%
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
IMG_SHAPE = IMG_SIZE + (3,)
conv_base = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=False,
    input_shape=IMG_SHAPE)

# %%
image_batch, label_batch = next(iter(train_dataset))
feature_batch = conv_base(image_batch)
print(feature_batch.shape)  # (32, 5, 5, 1280)
conv_base.trainable = False
# %%
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)  # (32, 1280)
# %%
prediction_layer = tf.keras.layers.Dense(1) 
prediction_batch = prediction_layer(feature_batch_average) 
print(prediction_batch.shape) # (32, 1)
# %%
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = conv_base(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x) 
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
# %%
model.summary()

# %%
h = model.fit(train_dataset, epochs=10,
validation_data=validation_dataset) 
subplot(121);plot(h.epoch,h.history["loss"]); plot(h.epoch,h.history["val_loss"]);legend(["loss","val_loss"]) 
subplot(122);plot(h.epoch,h.history["accuracy"]); plot(h.epoch,h.history["val_accuracy"]);legend(["acc","val_acc"])
# %%
print(len(conv_base.layers)) # 154
# Let's unfreeze the base model from layer 100 onwards. base_model.trainable = True
for layer in conv_base.layers[:100]:
    layer.trainable = False
# Let's compile the model again with a smaller learning rate: 
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5), metrics=['accuracy'])
# %%
import numpy as np
h2 = model.fit(train_dataset, epochs=20, initial_epoch=10, validation_data=validation_dataset)
# We can combine the results of both trainings as follows.
epoch = np.concatenate([h.epoch, h2.epoch])
subplot(121); plot(epoch,np.concatenate([h.history["loss"], h2.history["loss"]])) 
plot(epoch,np.concatenate([h.history["val_loss"], h2.history["val_loss"]])) 
legend(["loss","val_loss"])
subplot(122);
plot(epoch,np.concatenate([h.history["accuracy"], h2.history["accuracy"]])) 
plot(epoch,np.concatenate([h.history["val_accuracy"], h2.history["val_accuracy"]]))
legend(["acc","val_acc"])
# %%
# We can evaluate the trained model as follows.
loss, accuracy = model.evaluate(test_dataset) 
print('Test accuracy :', accuracy)
# The trained model can make predictions as follows.
# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next() 
predictions = model.predict_on_batch(image_batch).flatten()
# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1) 
print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)
figure(figsize=(10, 10))
for i in range(9):
    subplot(3, 3, i + 1)
    imshow(image_batch[i].astype("uint8"))
    title(class_names[predictions[i]]);axis("off")
