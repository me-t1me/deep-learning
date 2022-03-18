# %%
# TODO import dependancy
from logging import log
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from tensorflow.keras import activations
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math
# %%
# TODO creating data learning activation function
A = tf.cast(tf.range(-10, 10), dtype=tf.float32)

# creating a sigmoid function
# ! sigmoid function create curve lines, range (0,1)


def sigmoid(x):
    """
    gives 1 / (1 + exp(-x))
    """
    result = 1 / (1 + tf.exp(-x))
    return result


plt.plot(sigmoid(A), label="sigmoid", c="b")

# creating relu function
# ! relu function creates linear lines


def relu(x):
    return tf.maximum(0, x)


plt.plot(relu(A), label="relu", c="r")
# creating tanh dunction
# ! tanh also create curve line but its range is (-1,1)


def tanh(x):
    return tf.tanh(x)


plt.plot(tanh(A), label="tanh", c="g")
plt.legend()

# %%
# TODO create data set as made in previous module
X, y = make_circles(1000,
                    noise=0.03,
                    random_state=42
                    )
# visualizing data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# creating training and testing data
X_train, y_train = X[:800, :], y[:800]
X_test, y_test = X[800:, :], y[800:]
# %%
# plot decision boundary
# TODO creating function


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(x_in)
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


# %%
# TODO creating model
# set random seed
tf.random.set_seed(42)
# create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
])
# compile model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])
# fit model
history = model.fit(X_train, y_train, epochs=25)
# evaluate model
model.evaluate(X_test, y_test)
# visualizing prediction
plot_decision_boundary(model=model, X=X_test, y=y_test)
plot_decision_boundary(model=model, X=X_train, y=y_train)
# visualizing training
history_pd = pd.DataFrame(history.history)
pd.DataFrame(history_pd).plot()
# visualizing model
model.summary()
# %%
# create a new model
# set random seed
tf.random.set_seed(42)
# build model_2
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# compile model_2
model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
# create a callback
lr_schedular = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-4 * 10**(epochs/20))  # ? this is used to find ideal learning rate
# fit model_2
history_2 = model_2.fit(X_train, y_train, epochs=100, callbacks=[lr_schedular])

# ? this whole thing done to find ideal learning rate
lrs = 1e-4 * (10**(tf.range(0, 100)/20))
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history_2.history["loss"])
plt.xlabel("learning rate")
plt.ylabel("loss")
plt.show()

# %%
# TODO createing model with tr = 0.02(using graph loss vs log(lr))
# set random seed
tf.random.set_seed(42)
# build model_3
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# compile the model_3
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(lr=0.02),
                metrics=["accuracy"])
# fit the model_3
history_3 = model_3.fit(X_train, y_train, epochs=20)
# ? Evaluating model_3
# evaluate model_3
model_3.evaluate(X_test, y_test)

# visualize pridictions
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1, 2, 2)
plot_decision_boundary(model_3, X_test, y_test)

loss, accuracy = model_3.evaluate(X_test, y_test)
print(f"model loss : {loss}")
print(f"model accuracy : {(accuracy*100):.2f}%")

# ploting confusion matrix
y_pred = model_3.predict(X_test)
cm = confusion_matrix(y_test, tf.cast(tf.math.round(y_pred), dtype=tf.int32))

# ploting confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, )
disp.plot(cmap="RdYlBu")

# %%
