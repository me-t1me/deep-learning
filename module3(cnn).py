# 1. become one with data (visualize , visualize, visualize)
# 2. preprocess data (normalize)
# 3. create a model (start with baseline)
# 4. fit the model
# 5. evaluate the model
# 6. adjust the hyper parameters of model (experiment, experiment, experiment)
# 7. repeat until satisfied

# ! fitting a machine learning model comes in 3 steps
# ? 0. create a baseline model
# ? 1. beat the baseline model by overfitting the model
# ? 2. reduce overfitting

# ! way to reduce overfitting
# ? * increase number of conv layer
# ? * increase number of conv filters
# ? * add another dense layers to output of our flattened layers (increase units if it is not output layers)

# ! reducing overfitting
# ? * (giving model hard exercise) add data augmentation (testing data should not be augmented but should be rescaled)
# ? * add regularization layers (MAxpool2D)
# ? * add more data
# %%
# TODO importing dependencies
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.image as mpimg
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Activation
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.python.keras import activations
import wget
# %%
# unziping file
zip_ref = ZipFile("pizza_steak.zip", 'r')
zip_ref.extractall()
zip_ref.close()
# %%
# ? becoming one with data
# understanding with is in pizza_steak dir
for dirpath, dirnames, filenames in os.walk("pizza_steak"):
    print(
        f"There are {len(dirnames)} Directories and {len(filenames)} images in {dirpath}")
# %%
# TODO creating classes
data_dir = Path("pizza_steak/train")
class_name = np.array(os.listdir(data_dir))
class_name = class_name[1:]

# %%
# TODO visualizing data using function


def view_random_image(target_dir, target_class):
    path = target_dir + "/" + target_class
    images = os.listdir(path)
    image_name = random.choice(images)
    path_image = path + "/" + image_name
    image = mpimg.imread(path_image)
    plt.imshow(image)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {image.shape}")
    return image


img = view_random_image(target_dir="pizza_steak/train",
                        target_class="pizza")
# %%
# set random speed
tf.random.set_seed(42)
# preprocess the data (normalizing it)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
# importing the data
train_data = train_datagen.flow_from_directory(directory="pizza_steak/train",
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                               seed=42)
valid_data = valid_datagen.flow_from_directory(directory="pizza_steak/test",
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                               seed=42)
# %%
# building model_1
model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# compiling model_1
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
history_1 = model_1.fit(train_data,
                        steps_per_epoch=len(train_data),
                        epochs=5,
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
# %%
# creating non-cnn model
# set random seed
tf.random.set_seed(42)
# building model_non
model_non = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# compiling the model
model_non.compile(
    oss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])
# fiting the model_non
history_non = model_non.fit(train_data,
                            epochs=5,
                            steps_per_epoch=len(train_data),
                            validation_data=valid_data,
                            validation_steps=len(valid_data))
# %%
# set seed
tf.random.set_seed(42)
# build model_non_1
model_non_1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# compiling the model_non_1
model_non_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])
# fitting the model_non_1
history_non_1 = model_non_1.fit(train_data,
                                epochs=5,
                                steps_per_epoch=len(train_data),
                                validation_data=valid_data,
                                validation_steps=len(valid_data))

# %%
# 1. visualizing data
plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("pizza_steak/train", "steak")
plt.subplot(1, 2, 2)
pizza_img = view_random_image("pizza_steak/train", "pizza")
# %%
# 2. preprocessing data for model (batch is number to input network fits before calculating loss and optimizing itself)
train_dir = "pizza_steak/train"
test_dir = "pizza_steak/test"
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1/255.)
# importing images
train_data = train_gen.flow_from_directory(train_dir,
                                           target_size=(224, 224),
                                           class_mode='binary',
                                           batch_size=32)
test_data = test_gen.flow_from_directory(test_dir,
                                         target_size=(224, 224),
                                         class_mode='binary',
                                         batch_size=32,
                                         )

# %%
# filters -> matrix of weights
# kernel_size -> dimension of matrix
# strides -> how much pixels matrix moves
# padding -> if valid there will be padding to preserve border info (it also compress the edges)
# 3. creating baseline model
tf.random.set_seed(42)
model_baseline = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           strides=1,
           padding="valid",
           input_shape=(224, 224, 3),
           activation="relu"),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    Flatten(),
    Dense(1, activation="sigmoid")
])

# 4. compile the model_baseline
model_baseline.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                       optimizer=Adam(),
                       metrics=["accuracy"])
# 4. fit the model_baseline
history_baseline = model_baseline.fit(train_data,
                                      steps_per_epoch=len(train_data),
                                      epochs=5,
                                      validation_data=test_data,
                                      validation_steps=len(test_data)
                                      )


# %%
# 5. evaluating training curve (using 2 methods)
pd.DataFrame(history_baseline.history).plot(figsize=(10, 7))

# %%
# TODO creating loss ploting curve


def plot_loss_curve(history):
    """
    takes history of fit instance and plot loss function
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(loss))

    plt.figure(figsize=(10, 7))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(epochs, accuracy, label="Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.legend()
    plt.show()


# %%
# ! model_baseline is overfitting as training loss is decreasing and validation loss is increasing
plot_loss_curve(history_baseline)
# %%
# new baseline (with maxpool2d)
tf.random.set_seed(42)
# building new model_5
model_5 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(2),
    Flatten(),
    Dense(1, activation="sigmoid")
])
# compiling model_5
model_5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=Adam(),
                metrics=["accuracy"])
# fitting the model_5
history_5 = model_5.fit(train_data,
                        steps_per_epoch=len(train_data),
                        epochs=5,
                        validation_data=test_data,
                        validation_steps=len(test_data))

# visualize using model summary and see losses and accuracy
model_5.summary()
plot_loss_curve(history_5)  # ? now overfitting is reduced drastically
# %%
train_datagen_augmented = ImageDataGenerator(rescale=1./255,
                                             rotation_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)
train_dir = "pizza_steak/train"
test_dir = "pizza_steak/test"
print("Training data augmented: ")
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(
                                                                       224, 224),
                                                                   class_mode="binary",
                                                                   shuffle=False)
print("Training data non-augmented: ")
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                               shuffle=False)
print("Testing data non-augmented: ")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             class_mode='binary')
# %%
images, labels = train_data.next()
augmented_images, augmented_labels = train_data_augmented.next()

# %%
random_number = random.randint(0, 31)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(images[random_number])
plt.title("same")
plt.subplot(1, 2, 2)
plt.imshow(augmented_images[random_number])
plt.title("augmented")
# %%
# create the model
tf.random.set_seed(42)
model_6 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])
# compile model_6
model_6.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=Adam(),
                metrics=["accuracy"])
# fit model_6
history_6 = model_6.fit(train_data_augmented,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=test_data,
                        validation_steps=len(test_data))

plot_loss_curve(history_6)

# %%
# TODO shuffle our augmented training data
train_data_augmented_shuffle = train_datagen_augmented.flow_from_directory(train_dir,
                                                                           target_size=(
                                                                               224, 224),
                                                                           class_mode="binary",
                                                                           shuffle=True,
                                                                           seed=42)

# %%
# create model_7
tf.random.set_seed(42)
model_7 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])
# compiling model_7
model_7.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=Adam(),
                metrics=["accuracy"])
# fit model_7
history_7 = model_7.fit(train_data_augmented_shuffle,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented_shuffle),
                        validation_data=test_data,
                        validation_steps=len(test_data))

plot_loss_curve(history_7)

# %%
# ! what is good now-
# ? activation function
# ? optimization function
# ? learning rate

# ! what can we change -
# ? adding more layers
# ? increasing number of filters
# ? fitting for longer
# ? fitting more data
# %%
# TODO downloading costum picture for prdiction
wget.download(
    "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg")
steak = mpimg.imread("03-steak.jpeg", format='jpeg')
plt.imshow(steak)
plt.axis(False)
steak.shape  # shape = (4032, 3024, 3)

# %%
# TODO creating a function which can import the image, resize in way which can be used in predictions


def load_and_prep_image(filename, image_size=224):
    # read the file
    img = tf.io.read_file(filename=filename)
    # decode the file into image
    img = tf.image.decode_image(img)
    # resize the file
    img = tf.image.resize(img, size=[image_size, image_size])
    # rescaling the file
    img = img / 255.
    return img


# %%
steak = load_and_prep_image("03-steak.jpeg")
pred = model_7.predict(tf.expand_dims(steak, axis=0))
# %%
# TODO creating a function for visualizing image and getting prediction
pred_class = class_name[int(tf.round(pred))]
pred_class
# %%


def pred_and_plot(model, filename, class_name=class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        # if more than one output, take the max
        pred_class = class_names[pred.argmax()]
    else:
        # if only one output, round
        pred_class = class_names[int(tf.round(pred)[0][0])]

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


# %%
pred_and_plot(model_7, "03-steak.jpeg")
wget.download(
    "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-pizza-dad.jpeg")

pred_and_plot(model_7, "03-pizza-dad.jpeg")

# %%
# ! MUlticlass Image Classification
# ? 1. Become one with data (visualize! visualize! visualize!)
# ? 2. preprocess the data
# ? 3. create a baseline model
# ? 4. fit the model
# ? 5. evaluate the model
# ? 6. improve the model (1. overfit the model, 2. then reduce ovefitting)
# ? 7. Repeat until satisfied (Experiment! Experiment! Experiment!)
# %%
# unzipping file
zip = ZipFile("10_food_classes_all_data.zip", 'r')
zip.extractall()
zip.close()
# %%
train_dir = "10_food_classes_all_data/train"
test_dir = "10_food_classes_all_data/test"
class_names = os.listdir(train_dir)

# %%
# TODO 1. becoming one with data
img = view_random_image(train_dir, target_class=random.choice(class_names))
plt.figure(figsize=(10, 7))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(view_random_image(
        train_dir, target_class=random.choice(class_names)))
    plt.axis(False)
# %%
# TODO 2. preprocessing data
train_datagen = ImageDataGenerator(rescale=1/225.)
test_datagen = ImageDataGenerator(rescale=1/225.)
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='categorical',
                                             seed=42)

# %%
# TODO creating a baseline model && fitting model
# creating baseline model
model_8 = Sequential([
    Conv2D(10, 3, input_shape=(224, 224, 3)),
    Activation(activation="relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax")
])
# compiling model_8
model_8.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=Adam(),
                metrics=["accuracy"])

# fitting model_8
history_8 = model_8.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

# %%
# TODO Evaluate the model
model_8.evaluate(test_data)
pd.DataFrame(history_8.history)["loss"].plot()
pd.DataFrame(history_8.history)["val_loss"].plot()
plt.legend()
# %%
# TODO improving the model
# ! model is currently overfitting, so to fix this
# ? get more data
# ? simplify the model -> overfitting indicates that model is learning too way to reduce its complexicty
# ? use data augmentation
# ? transfer learning
model_9 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax")
])
# compile model_9
model_9.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=Adam(),
                metrics=["accuracy"])
# %%
# fitting the model_9
history_9 = model_9.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))
# %%
# evaluating model_9
pd.DataFrame(history_9.history)["loss"].plot()
pd.DataFrame(history_9.history)["val_loss"].plot()
plt.legend()
# %%
# improving model by augmenting data
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,
                                             zoom_range=0.2,
                                             height_shift_range=0.2,
                                             width_shift_range=0.2,
                                             horizontal_flip=True,
                                             )

train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(
                                                                       224, 224),
                                                                   class_mode='categorical',
                                                                   batch_size=32,
                                                                   seed=42)
# %%
tf.random.set_seed(42)
# creating model_10
model_10 = tf.keras.models.clone_model(model_9)
# compile model_10
model_10.compile(loss="categorical_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

# fitting the model_10
history_10 = model_10.fit(train_data_augmented,
                          epochs=5,
                          steps_per_epoch=len(train_data_augmented),
                          validation_data=test_data,
                          validation_steps=len(test_data))

# %%
# getting for prediction Images
wget.download(
    "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-pizza-dad.jpeg")
wget.download(
    "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-hamburger.jpeg")
wget.download(
    "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-sushi.jpeg")
wget.download(
    "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg")

# %%
# ! this model made is very bad at predicting output
pred_and_plot(model=model_10,
              filename="03-hamburger.jpeg",
              class_name=class_names)

# %%
# saving and loading model
model_10.save("saved_trained_model_10")
# %%
loaded_model = tf.keras.models.load_model("saved_trained_model_10")
# %%
loaded_model.summary()
# %%
