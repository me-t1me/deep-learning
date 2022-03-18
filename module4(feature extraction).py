
# ! How to find best model from tensorflow hub
# ? 1. go to paperwithcode and find papers related to ur problem
# ? 2. U will get what type od architecture to use (google them to become fimilar with it)
# ? 3. got to tensorflowHub and search accordingly
# %%
# ! there are 3 types of transfer learning
# ? "As is" (not changing output layer as well) meaning not changing any thing
# ? "Feature Extraction" (dataset and output layer changes) meaning keeping all other layers same and changing only output layer
# ? "Fine tuning" (unfreeze some or all layers and dataset can change but output layer remain same) meaning fine-tuning some or all layers of network excluding output layer
# %%
import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
# %%
# TODO becoming one with data
# unzipping the data
zip_file = zipfile.ZipFile("10_food_classes_10_percent.zip")
zip_file.extractall()
zip_file.close()
# %%
# how many image are in these folder
for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
    print(
        f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")

# %%
# ! hyperperameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

print("Training images: ")
train_data_10_percent = train_datagen.flow_from_directory(train_dir,
                                                          target_size=IMAGE_SIZE,
                                                          class_mode='categorical',
                                                          batch_size=BATCH_SIZE)

print("Testing images: ")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=IMAGE_SIZE,
                                             batch_size=BATCH_SIZE,
                                             class_mode='categorical')
# %%
# ! famous callbacks
# ? traching experiment with Tensorboard Callback
# ? creating model checkpoint with ModelCheckpoint Callback
# ? Stopping model before ovefitting with EarlyStopping Callback
# %%
# creating tensorboard callback


def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + '/' + experiment_name + '/' + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log file to: {len(log_dir)}")
    return tensorboard_callback


# %%
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

# %%
# creating a model builder function


def create_model(model_url, num_classes=10):
    """
    Takes a tensorflow model url and creates a keras sequential model with it.

    Args:
        model_url (str): A Tensorflow Hub feature extraction URL.
        num_classes (int): Number of output neuron on output layer,
            should be equal to number of target classes, defualt 10

    Returns:
        An uncompiled Keras Layers model with model_url as feature extractor
        layer and Dense layer with num_classes output neurons.
    """
    # downloading model from tensorflow hub
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name="feature_extractor_layer",
                                             input_shape=IMAGE_SIZE + (3,))

    # creating the model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model


# %%
# creating ResNet model
resnet_model = create_model(model_url=resnet_url,
                            num_classes=train_data_10_percent.num_classes)
# compiling resnet_model
resnet_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])
# fitting resnet_model
history_resnet_model = resnet_model.fit(train_data_10_percent,
                                        steps_per_epoch=len(
                                            train_data_10_percent),
                                        epochs=5,
                                        validation_data=test_data,
                                        validation_steps=len(test_data),
                                        callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                               experiment_name="resnet50V2")])
# %%
# creating plot_loss_function to plot losses of models


def plot_loss_curves(history):
    # creating a array of history
    arr = history.history
    # creating variables
    loss = arr["loss"]
    val_loss = arr["val_loss"]

    accuracy = arr["accuracy"]
    val_accuracy = arr["val_accuracy"]

    epochs = range(len(arr["loss"]))
    # ploting graphs
    plt.figure()
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.show()

    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.show()


# %%
plot_loss_curves(history_resnet_model)
# %%
# creating efficientnet model
efficientnet_model = create_model(model_url=efficientnet_url,
                                  num_classes=train_data_10_percent.num_classes)
# compiling model
efficientnet_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=["accuracy"])
# fitting efficientnet model
efficientnet_history = efficientnet_model.fit(train_data_10_percent,
                                              steps_per_epoch=len(
                                                  train_data_10_percent),
                                              epochs=5,
                                              validation_data=test_data,
                                              validation_steps=len(test_data),
                                              callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                                     experiment_name="efficientnetb0")])

# %%
plot_loss_curves(efficientnet_history)
# %%
# ! use code below to upload to tensorboard (use bash terminal)
# tensorboard dev upload --logdir ./tensorflow_hub/ \
#   --name "EfficientNetB0 vs. ResNet50V2" \
#   --description "Comparing two different TF Hub feature extraction models architectures using 10% of training images" \
#   --one_shot
# ! how to view experiment on terminal
# tensorboard dev list
# tensorboard dev delete --experiment_id VhaTo7FNTrm1OnMnplLwZQ
# %%
