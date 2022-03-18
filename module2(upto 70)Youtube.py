# ! think about what input is i.e. it is image or numbers
# ! 1. how computer inpret images (color channels)
# ! 2. how humans interpert images (colors)
# ! 3. find common things to use as tensor
# %%
# installing modules
from numpy.core.fromnumeric import shape
import tensorflow as tf
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# %%
# DATA
n_sample = 1000
X, y = make_circles(
    n_samples=n_sample,
    noise=0.03,
    random_state=42
)
# %%
# visualizing our data
circle = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# %%
# set random seed
tf.random.set_seed(42)
# build the model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
# compile the model
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
# fit the model
model_1.fit(X, y, epochs=5)
# increase numbers of run
model_1.fit(X, y, epochs=200, verbose=0)
model_1.evaluate(X, y)
# %%
# increase hidden layers
# set random seeds
tf.random.set_seed(42)
# build model_2
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1),
])
# compile model_2
model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
# fit model_2
model_2.fit(X, y, epochs=100)
# evaluating model_2
model_2.evaluate(X, y)
# %%
# set the random seed
tf.random.set_seed(42)
# create the model_3
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
# compile the model
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
# fit the model
# model_3.fit(X, y, epochs=100)
# evaluate model_3
# model_3.evaluate(X, y)

# %%
# predictions model_3
y_pred = model_3.predict(X)
# %%
# create ploting function (visualize prdiction)


def plot_decision_boundary(model_3, X, y):
    """
    plots the decision boundary created by amodel predicting on X.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model_3.predict(x_in)
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
plot_decision_boundary(model_3, X, y)

# %%
# create a regression data to see model_3 compatibilty with it
tf.random.set_seed(42)
# create data
X_regression = tf.range(0, 1000, 5)
Y_regression = tf.range(100, 1100, 5)

# %%
# create train and test data
X_train = X_regression[:150]
X_test = X_regression[150:]
Y_train = Y_regression[:150]
Y_test = Y_regression[150:]

# %%
# set random seed
tf.random.set_seed(42)
# build model_3 again
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
# compile model_3
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])
# fit model_3
model_3.fit(X_train, Y_train, epochs=100)
# %%
Y_reg_pred = model_3.predict(X_test)
# %%
plt.figure(figsize=(10, 7))
plt.scatter(X_train, Y_train, c="b", label="Training Data")
plt.scatter(X_test, Y_test, c="g", label="Testing Data")
plt.scatter(X_test, Y_reg_pred, c="r", label="Prediction")
plt.legend()
plt.show()

# %%
# set random seed
tf.random.set_seed(42)
# create model_4
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
])
# compile the model_4
model_4.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
# fit the model_4
model_4.fit(X, y, epochs=100)
# visualize model_4
plot_decision_boundary(model_4, X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# %%
# creating model using playgroud
# set random seed
tf.random.set_seed(42)
# build model_5
model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu,),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    # in playground tensorflow they don't show output layer
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# compile model_5
model_5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])
# fit the model_5
model_5.fit(X, y, epochs=100)
plot_decision_boundary(model_5, X, y)
model_5.evaluate(X, y)

# %%
# ! I can get any shape with linear(relu) and non linear(sigmoid) funcitons
