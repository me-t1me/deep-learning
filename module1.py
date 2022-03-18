# %%
# what we are going to do -
# architecture of neural Network
# input and output shape of regression models
# creating custom data to view and fit
# step in modeling:
#     creating a model
#     compiling a model
#     fitting ...
#     evaluating ...
# different evaluating methods
# saving and loading models
# %%
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import tensorflow as tf
# %%
# creating data
import numpy as np
import matplotlib.pyplot as plt

X = np.array([-7., -4., -1., 2., 5., 8., 11., 14., ])
Y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])
plt.scatter(X, Y)

# input and output shape
# creating tensors for house pricing problem
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
house_info, house_price
# %%
# creating tensors for X and Y
X = tf.constant(X, dtype=tf.float32)
Y = tf.constant(Y, dtype=tf.float32)
X[0].shape, Y[0].shape  # these are scaler value so no shape
plt.scatter(X, Y)
# %%
# modelling -
# 1. creating a model -> defining layers such as input, hidden and output
# 2. compiling -> loss function (this function tell out model where it is wrong), optimizer (tell our model how to improve) and evalution metrics (what we can use to interpret the performace of our model)
# 3. fitting a model -> letting model find pattern between X and Y

# set randon seed, so there can be reproducbilty
tf.random.set_seed(42)

# %%
# 1. creating a model using sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
# 2. compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(), metrics=["mae"])  # function such as mae and sgd can be written in string as well as regular methord
# 3. fit the model
model.fit(X, Y, epochs=5)
# making a prediction
y_pred = model.predict([17])
y_pred
# %%
# how to improve our model
# we can improve our model, by altering the steps we took to create a model
# 1. creating a model: - add more layer, increase number of hidden units(neurons) with each of hidden layers, change activation function of each layer
# 2. compiling a model: - change the optimization function, learning rate of said optimization function.
# 3. fitting a model: - more epochs or more data (give our model more example to learn from)

# change one thing at a time, so that it takes less time and but improve
# %%
# lets rebuild model
# creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
# compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

# fit the model (changing epochs this time)
model.fit(X, Y, epochs=100)
# %%
model.predict([17.0])

# %%
# rebulding model by myself
# creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
# compile the model (change the optimizer to adam)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(lr=0.01),
              metrics=['mae'])
# fitting the model
model.fit(X, Y, epochs=100)
model.predict([17.0])  # mae is worse and prdicted number is also worse
# changing optimizer doesnot work
# %%
tf.random.set_seed(42)
# rebuilding the model
# create the model (with extra hidden layer with 100 nuerons and activation)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation=None),
    tf.keras.layers.Dense(1)
])
# compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])
# fitting the model
model.fit(X, Y, epochs=100)
# 31.38 so it is worse then before even though mae is less
model.predict([17.])
# %%
# rebuilding the model
# create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation=None),
    tf.keras.layers.Dense(1)
])
# compile the model (chaning optimizer and lr)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(lr=0.01),
              metrics=['mae'])
# fitting the model
model.fit(X, Y, epochs=100)
# 31.38 so it is worse then before even though mae is less
model.predict([17.])
# mse = 0.1814 and prediction value = 27.21
# this is much better
# %%
# evaluting the model (data not seen by model)
# build a model -> fit it -> evaluate it -> tweak a model -> fit it -> evaluate it -> tweak a model
# for creating model: experiment, experiment, experiment
# for evaluating model: visualize, visualize, visualize
# what to visualize -
# 1. the data - what data are we working with? what does it look like
# 2. the model itself - what does our model look like
# 3. training of model - how does model perform while it learns
# 4. the prediction of model - comparing prediction(new data) and real data (we tried it)
# %%
# making a bigger dataset
X = tf.range(-100, 100, 4)
X
# lebels for the dataset
Y = X + 10
Y
# visualize the data (evaluate(visualizing point 1))
plt.scatter(X, Y)
# %%
# three set (for this purpose 2 set) ->
# training set - the model learn from this data, which os 80%
# validation set - the model get tuned on this data (data used for tweaking), 10-15% (this is usally not used when we have less data)
# test set - the model gets evaluted on this data to test what is had learned, this set is 10-15%

X_train = X[:int(0.8*50)]
Y_train = Y[:40]
X_test = X[int(0.8*50):]
Y_test = Y[40:]
len(X_train), len(X_test), len(Y_train), len(Y_test)
# %%
# lets visualize it as before (because there is 4 sets)
plt.figure(figsize=(10, 7))
# plot training in blue
plt.scatter(X_train, Y_train, c="b", label="Training data")
# plot test data in green
plt.scatter(X_test, Y_test, c="g", label="Testing data")
# show a legend
plt.legend()
# %%
# create a model
model = tf.keras.Sequential(
    tf.keras.layers.Dense(1)
)
# compile the model
model.compile(loss=tf.keras.losses.mse,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])
# # fitting the model
# model.fit(X_train, Y_train, epochs=100) # fit on training data

# visualizing the model
model.summary()  # this gives error, so create a model which have input shape in layer
# %%
# for reproducibilty
tf.random.set_seed(42)
# create a model (same as before)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, input_shape=(1,), name="input_layer"),
    tf.keras.layers.Dense(units=1, name="output_layer")
], name="model_1")
# compile a model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"]
              )

model.summary()
# params -> are weights which comes from input + bias
# total params -> total no. of parameters in the model
# trainable params -> parameters model will train oo
# non-trainable params -> model has already trained on these params

# let fit model to training data
model.fit(X_train, Y_train, epochs=100, verbose=0)
# call summary again
model.summary()
# visualizing model
plot_model(model, show_shapes=True)
# %%
# visualing our model predictions
# y_pred is truth
Y_pred = model.predict(X_test)
Y_pred
# %%
Y_test
# %%
# to visualize model pridiction -> u do Y_pred vs Y_test
# we will create a ploting function for predictions


def plot_prediction(train_data=X_train,
                    train_labels=Y_train,
                    test_data=X_test,
                    test_labels=Y_test,
                    predictions=Y_pred):
    """[summary]

    Args:
        train_data ([type], optional): [Training data]. Defaults to X_train.
        train_labels ([type], optional): [Training labels]. Defaults to Y_train.
        test_data ([type], optional): [Testing data]. Defaults to X_test.
        test_labels ([type], optional): [Testing labels]. Defaults to Y_test.
        predictions ([type], optional): [Prediction of model]. Defaults to Y_pred.
    """
    plt.figure(figsize=(10, 7))
    # plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # plot testing data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # plot prediction in red
    plt.scatter(X_test, predictions, c="r", label="Prediction")
    # show legend
    plt.legend()


# calling function to visualize these things
plot_prediction(train_data=X_train,
                train_labels=Y_train,
                test_data=X_test,
                test_labels=Y_test,
                predictions=Y_pred)
# prediction graph can look but if scale is big it will problems
plot_prediction()

# %%
# how to evaluate model other then visualizing
# 1. MAE -> for regression model
# 2. MSE -> when larger error are significant then smaller error (if 100 off is far more worsed then 10 off)
# 3. Huber -> combination of both

model.evaluate(X_test, Y_test,)

# finding mae through different formula
mae = tf.keras.losses.MAE(Y_test, Y_pred)
tensor_pred = tf.squeeze(tf.constant(Y_pred))
tf.keras.losses.MAE(Y_test, tensor_pred)
tf.keras.losses.MSE(Y_test, tensor_pred)

# make fuction reuse mae and mse


def mae(Y_true, Y_pred):
    return tf.metrics.mean_absolute_error(Y_true, tf.squeeze(Y_pred))


def mse(Y_true, Y_pred):
    return tf.metrics.mean_squared_error(Y_true, tf.squeeze(Y_pred))


mae(Y_test, tensor_pred), mse(Y_test, tensor_pred)
# %%
# running experiment to improve our model
# build -> fit -> evalute it -> tweak it -> fit it -> evaluate

# lets do 3 modelling experiment -
# 1. 1 layer, 100 epochs
# 2. 2 layer, 100 epochs
# 3. 2 layer, 500 epochs

# build model_1
tf.random.set_seed(42)

# creating model_1
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
# compile model_1
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])
# fit model_1
model_1.fit(X_train, Y_train, epochs=100)
# plot prediction model_1
Y_pred_1 = model_1.predict(X_test)
Y_pred_1
plot_prediction(predictions=Y_pred_1)
# calucation metrics
mae_1 = mae(Y_test, Y_pred_1)
mse_1 = mse(Y_test, Y_pred_1)
mae_1, mse_1
# %%
# build model_2 (2 layer, 100 epochs)
tf.random.set_seed(42)

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
# compile model_2
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mse'])
# fit the model_2
model_2.fit(X_train, Y_train, epochs=100)
Y_pred_2 = model_2.predict(X_test)
# plot predictions of model_2
plot_prediction(predictions=Y_pred_2)
# metrics model_2
mae_2 = mae(Y_test, Y_pred_2)
mse_2 = mse(Y_test, Y_pred_2)
# %%
# set random
tf.random.set_seed(42)
# build model_3
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
# compile model_3
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])
# fit model_3
model_3.fit(X_train, Y_train, epochs=500)
# plot predictions
Y_pred_3 = model_3.predict(X_test)
plot_prediction(predictions=Y_pred_3)
# metrics evaluation
mae_3 = mae(Y_test, Y_pred_3)
mse_3 = mse(Y_test, Y_pred_3)
# %%
# comparing all models
model_results = [
    ["model_1", mae_1.numpy(), mse_1.numpy()],
    ["model_2", mae_2.numpy(), mse_2.numpy()],
    ["model_3", mae_3.numpy(), mse_3.numpy()]
]
all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
all_results
# model_2 is best
model_2.summary()
# %%
# saving our models
# 1. savedmodel format
model_2.save("My_model_1")
# 2. HDF5 format
model_2.save("My_model_1_HDF5_format.h5")
# savedformat
loaded_My_model_1 = tf.keras.models.load_model('My_model_1')
loaded_Y_pred = loaded_My_model_1.predict(X_test)
loaded_Y_pred == Y_pred_2
# HDF5
loaded_My_model_1_HDF5_format = tf.keras.models.load_model(
    'My_model_1_HDF5_format.h5')
loaded_Y_pred_H = loaded_My_model_1_HDF5_format.predict(X_test)
loaded_Y_pred_H == Y_pred_2
# %%
# * larger data
insurance = pd.read_csv(
    'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
# %%
# there are female and male as input, lets change it using one hot encoding
insurance_one_hot = pd.get_dummies(insurance)

# %%
insurance_one_hot

# %%
# ? create X and Y
X = insurance_one_hot.drop("charges", axis=1)
Y = insurance_one_hot["charges"]
# %%
# converting into test and train set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# %%
# modeling
# set random seed
tf.random.set_seed(42)
# build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
# compile model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])
# fit the model
model.fit(X_train, Y_train, epochs=100)
# %%
model.evaluate(X_test, Y_test)
# %%
# set random seed
tf.random.set_seed(42)
# build model_2
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])
# compile the model
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])
# fit the model
model_2.fit(X_train, Y_train, epochs=100)

model_2.evaluate(X_test, Y_test)
# %%
# set random seed
tf.random.set_seed(42)
# # build model_3
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
# compile model_3
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])
# fit the model_3
history = model_3.fit(X_train, Y_train, epochs=200)
# %%
model_3.evaluate(X_test, Y_test)
# %%
model_2.evaluate(X_test, Y_test)
# %%
# plot the lost curve to see how much epochs should be use
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

# %%
# normalize and standardize
insurance
# %%
ct = make_column_transformer(
    (MinMaxScaler(), ['age', 'bmi', "children"]),
    (OneHotEncoder(handle_unknown='ignore'), ['sex', 'smoker', 'region'])
)
X = insurance.drop("charges", axis=1)
Y = insurance["charges"]
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# %%
ct.fit(X_train)
# %%
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)
X_train_normal.shape
# %%
# set random seed
tf.random.set_seed(42)
# build the model
model_insurance = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
# compiling model
model_insurance.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=['mae'])
# fit the model
model_insurance.fit(X_train_normal, Y_train, epochs=100)
# evaluate model
model_insurance.evaluate(X_test_normal, Y_test)

# %%
