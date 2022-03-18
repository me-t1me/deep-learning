# %%
# TODO importing dependency
from pandas.core.algorithms import mode
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tensorflow.python.ops.numpy_ops.np_array_ops import imag
# %%
# TODO importing data
(train_data, train_label), (test_data, test_label) = fashion_mnist.load_data()
# creating list
class_list = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# %%
# ? visualize data is important to become familier with it
plt.figure(figsize=(7, 7))
for i in range(4):
    plt.subplot(2, 2, i+1)
    ran_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[ran_index],
               cmap=plt.cm.binary)
    plt.title(class_list[train_label[ran_index]])
    plt.axis(False)

plt.show()
# %%
# ? if labels are one hot encoded use categoricalcrossentropy
# ? if labels are integer use sparsecategoricalcrossentropy
# building the model
# set random seed
tf.random.set_seed(42)
# create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
# compile model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
# fit model
non_norm_history = model.fit(
    train_data, tf.one_hot(train_label, depth=10), epochs=10, validation_data=(test_data, tf.one_hot(test_label, depth=10)))
# %%
# ? * nueral network prefer normalized numerical data with good shape
# normalizing data
train_data_norm = train_data / 255
test_data_norm = test_data / 255
# %%
# building model_norm
# set random seed
tf.random.set_seed(42)
# creating model_norm
model_norm = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
# compiling model_norm
model_norm.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=["accuracy"])
# fitting model_norm
norm_history = model_norm.fit(train_data_norm,
                              train_label,
                              epochs=10,
                              validation_data=(test_data_norm, test_label))

# %%
pd.DataFrame(non_norm_history.history).plot(title="non-normalized")
pd.DataFrame(norm_history.history).plot(title="normalized")
plt.show()

# %%
# TODO finding ideal learning rate
# set random seed
tf.random.set_seed(42)
# create model_2
model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
# compile model_2
model_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
# creating callback
lr_call = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3 * 10**(epoch / 20))
# fitting model_2
find_lr_history = model_2.fit(train_data_norm,
                              train_label,
                              epochs=40,
                              validation_data=(test_data_norm, test_label),
                              callbacks=[lr_call]
                              )
# ploting log of learning rate and loss
loss = pd.DataFrame(find_lr_history.history)['loss']
lrs = 1e-3 * 10**(tf.range(0, 40) / 20)
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, loss)
plt.xlabel('Log of lrs')
plt.ylabel('loss')
plt.show()
# ? find lowest point and go back a little bit (ideal lr = 0.001)
# %%
# find ideal epochs
# set random seed
tf.random.set_seed(42)
# build model_3
model_3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
# compile model
model_3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
# fit the model_3
history_3 = model_3.fit(train_data_norm,
                        train_label,
                        epochs=20,
                        validation_data=(test_data_norm, test_label))
# %%
# roundup because prediction are in percentage to convert into one hot encoding
test_pred = np.round(model_3.predict(test_data_norm))
# change one hot encoding to simple indexis
non_one_hot = tf.argmax(test_pred, axis=1)
#  making confusion matrix
cm = confusion_matrix(test_label, non_one_hot)
disp = ConfusionMatrixDisplay(cm, display_labels=class_list)
disp.plot(cmap=plt.cm.Blues)
# %%
# plot a random image


def plot_random_image(model, images, true_labels, classes):
    """
    Picks a random image, plot it and label it with a prediction and true label
    """
    # creating random integer
    i = random.randint(0, len(images))

    # creating image
    target_image = images[i]
    # true value of image
    true_value = true_labels[i]
    # predicted value of image
    pred_value = tf.argmax(
        model.predict(target_image.reshape(1, 28, 28)), axis=1).numpy()[0]
    # ploting randomly chosen image
    plt.imshow(target_image, cmap=plt.cm.binary)
    # change color depending on situation
    matching_per = tf.reduce_max(model.predict(
        target_image.reshape(1, 28, 28))).numpy()*100
    if pred_value == true_value:
        color = "green"
    else:
        color = "red"
    # adding xlabel
    plt.xlabel(
        f"Pred: {classes[pred_value]} {(matching_per):2.0f}% (True: {classes[true_value]})", color=color)


def rand_4_image():
    plt.figure(figsize=(15, 10))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plot_random_image(model_3,
                          test_data_norm,
                          test_label,
                          class_list)


# %%
rand_4_image()

# %%
# TODO how to get patterns from model
# extract one layer
model_3.layers[1]
# get the patterns from this layer
weights, biases = model_3.layers[2].get_weights()
weights, weights.shape, biases, biases.shape
# %%
