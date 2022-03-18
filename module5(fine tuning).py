# %%
# downloading helper_function.py from github
import wget
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from helper_functions import create_tensorboard_callback, plot_loss_curves, pred_and_plot, unzip_data, walk_through_dir
tf.random.set_seed(42)
# %%
wget.download(
    "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py")
# %%
walk_through_dir("10_food_classes_10_percent")
# %%
train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# %%
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            label_mode='categorical',
                                                                            image_size=IMG_SIZE,
                                                                            batch_size=BATCH_SIZE)

test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                batch_size=BATCH_SIZE)
# %%
# 1. create base_model with tf.keras.applications
base_model = tf.keras.applications.EfficientNetB0(include_top=False)

# 2. freezing layers in base_model
base_model.trainable = False

# 3. creating inputs into our model
inputs = tf.keras.layers.Input(shape=IMG_SIZE + (3,), name="input_layer")

# 5. pass the inputs to base_model
x = base_model(inputs)
print(f"shape after passing input to base_model = {x.shape}")

# 6. using global average pooling
x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling_layer")(x)
print(f"Shape after x to globalaveragepooling2d = {x.shape}")

# 7. creating output layers
outputs = tf.keras.layers.Dense(
    10, activation="softmax", name="output_layer")(x)

# 8. combining model_0
model_0 = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_model")

# 9. compiling the model_0
model_0.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 10. fitting the model_0
history_10_percent = model_0.fit(train_data_10_percent,
                                 steps_per_epoch=len(train_data_10_percent),
                                 epochs=5,
                                 validation_data=test_data,
                                 validation_steps=int(0.25 * len(test_data)),
                                 callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                                        experiment_name="10_percent")])

# %%
# evaluate model_0
model_0.evaluate(test_data)
# %%
# checking layers of base_model
for layer_number, layers in enumerate(base_model.layers):
    print(layer_number, layers.name)
# %%
# summary of base_model
base_model.summary()
# model_0 summary
model_0.summary()
# %%
plot_loss_curves(history_10_percent)
# %%
# understanding global maxpool2d
input_shape = (1, 2, 2, 4)
# creating input layer
input_layer = tf.random.normal(shape=input_shape)
print(f"input_Layer = {input_layer}")
# creating global maxpooling layer
global_maxpool_layer = tf.keras.layers.GlobalMaxPool2D()(input_layer)
print(f"GlobalMaxPooling2D = {global_maxpool_layer}")

# finding shape of both layer
print(f"shape of input_layer = {input_layer.shape}")
print(f"shape of global_maxpool_layer = {global_maxpool_layer.shape}")

# comparing it with reduce_max
# creatin reduce_max function
reduce_max_layer = tf.reduce_max(input_layer, axis=(1, 2))
print(f"reduce_max_layer = {reduce_max_layer}")

# %%
# TODO unziping new dataset
unzip_data("10_food_classes_1_percent.zip")
# %%
train_dir_1_percent = "10_food_classes_1_percent/train"
test_dir = "10_food_classes_1_percent/test"
walk_through_dir(train_dir_1_percent)
walk_through_dir(test_dir)

# %%
IMG_SIZE = (224, 224)
train_data_1_percent = image_dataset_from_directory(directory=train_dir_1_percent,
                                                    label_mode="categorical",
                                                    batch_size=32,
                                                    image_size=IMG_SIZE)
test_data = image_dataset_from_directory(directory=test_dir,
                                         label_mode="categorical",
                                         batch_size=32,
                                         image_size=IMG_SIZE)

# %%
# 1. creating data augmentation layer
data_augmentation = keras.Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2)
], name="data_augmentation")
# %%
# visualizing data
target_class = random.choice(train_data_1_percent.class_names)
target_dir = train_dir_1_percent + "/" + target_class
target_img = random.choice(os.listdir(target_dir))
target_img_path = target_dir + "/" + target_img
img = mpimg.imread(target_img_path)
# ploting it
print(f"Class: {target_class}")
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis(False)
plt.title("Before augmentation")
augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
plt.subplot(1, 2, 2)
plt.imshow(tf.cast(tf.squeeze(augmented_img, axis=0), dtype=tf.int32))
plt.axis(False)
plt.title("After augmentation")
# %%
# ! creating model_1
base_model = keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# creating input layer
inputs = layers.Input(shape=IMG_SIZE + (3,),
                      name="input_layers")

# appling inputs to augmentation
x = data_augmentation(inputs)

# now appling it to base model
x = base_model(x, training=False)

# creating globalaveragepooling2d layer to get feature vectors
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

# creating dense layers
outputs = layers.Dense(10, activation="softmax",
                       name="output_layers")(x)

# creating model_1
model_1 = keras.Model(inputs=inputs, outputs=outputs, name="model_1")

# compile model_1
model_1.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
# fit model_1
history_1_percent = model_1.fit(train_data_1_percent,
                                steps_per_epoch=len(train_data_1_percent),
                                epochs=5,
                                validation_data=test_data,
                                validation_steps=int(0.25 * len(test_data)),
                                callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                                       experiment_name="1_percent_data_aug")])
# %%
model_1.summary()
results_1_percent_data_aug = model_1.evaluate(test_data)
results_1_percent_data_aug
plot_loss_curves(history_1_percent)

# %%
train_dir_10_percent = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"
# %%
# data inputs
train_data_10_percent = keras.preprocessing.image_dataset_from_directory(train_dir_10_percent,
                                                                         label_mode='categorical',
                                                                         image_size=(224, 224))
test_data = keras.preprocessing.image_dataset_from_directory(test_dir,
                                                             label_mode='categorical',
                                                             image_size=(224, 224))

# %%
# creating augmented layer
data_augmentation = keras.Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomHeight(0.2)
], name="data_augmentation")
# %%
input_shape = (224, 224, 3)

# creating frozen base model
base_model = keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# creating all layers
inputs = layers.Input(shape=input_shape, name="input_layer")
x = data_augmentation(inputs)
x = base_model(x, training=False)
global_average_pool_layer = layers.GlobalAveragePooling2D(
    name="global_average_pooling_layer")(x)
outputs = layers.Dense(10, activation="softmax", name="output_layer")(
    global_average_pool_layer)

# creating model_2
model_2 = keras.Model(inputs=inputs, outputs=outputs)

# compile model_2
model_2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# %%
# creating a checkpoint callback
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_best_only=False,
                                                      save_weights_only=True,
                                                      save_freq="epoch",
                                                      verbose=1)

# %%
initial_epochs = 5
history_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                          epochs=initial_epochs,
                                          steps_per_epoch=len(
                                              train_data_10_percent),
                                          validation_data=test_data,
                                          validation_steps=int(
                                              0.25 * len(test_data)),
                                          callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                                                 experiment_name="10_percent_data_aug"),
                                                     checkpoint_callback])

# %%
result_10_percent_data_aug = model_2.evaluate(test_data)
plot_loss_curves(history_10_percent_data_aug)

# %%
# loading in checkpoint in model_2
model_2.load_weights(checkpoint_path)
loaded_weight_model_result = model_2.evaluate(test_data)
loaded_weight_model_result
# result_10_percent_data_aug == loaded_weight_model_result

# %%
model_2.layers
for layer in model_2.layers:
    print(layer.name, layer.trainable)
for i, layer in enumerate(model_2.layers[2].layers):
    print(i + 1, layer.name, layer.trainable)

# %%
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

# recompiling the model_2
model_2.compile(loss=keras.losses.CategoricalCrossentropy(),
                # TODO decrease learning rate by 10x for fine-tuning
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                metrics=['accuracy'])

for layer_number, layer in enumerate(model_2.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)

print(len(model_2.trainable_variables))
# %%
fine_tune_epochs = 5 + initial_epochs
# refit the model
history_fit_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                              epochs=fine_tune_epochs,
                                              steps_per_epoch=len(
                                                  train_data_10_percent),
                                              validation_data=test_data,
                                              validation_steps=int(
                                                  0.25 * len(test_data)),
                                              # ! meaning start training where left off by previous fit call
                                              initial_epoch=history_10_percent_data_aug.epoch[-1],
                                              callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                                                     experiment_name="10_percent_fine_tune_last_10")])

# %%
# evaluate the fine_tune model
results_fine_tune_10_percent = model_2.evaluate(test_data)
results_fine_tune_10_percent
result_10_percent_data_aug

# %%
# creating a function for comparing histories


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    comparing 2 tensorflow history object
    """
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_loss = original_history.history["val_loss"]
    val_acc = original_history.history["val_accuracy"]

    # combining both histories
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # making a plot
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training accuracy")
    plt.plot(total_val_acc, label="Validation accuracy")
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label="Start Fine-tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    # making plot 2
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label="Start Fine-tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

# %%
# since food 101 all data is already unziped so directly use it
train_dir = "10_food_classes_all_data/train"
test_dir = "10_food_classes_all_data/test"

# checking number of images
walk_through_dir(train_dir) 
walk_through_dir(test_dir)
# %%
# setting up data inputs
IMG_SIZE = (224,224)
train_data_10_classes_full = tf.keras.preprocessing.image_dataset_from_directory(directory = train_dir,
                                                                                 label_mode='categorical',
                                                                                 image_size=IMG_SIZE)

test_data = tf.keras.preprocessing.image_dataset_from_directory(directory = test_dir,
                                                                label_mode='categorical',
                                                                image_size=IMG_SIZE) 
# %%
# instead of finetuning model_2 we will load from checkpoint to see the difference between them
model_2.load_weights(checkpoint_path)

# lets verify it is reverted model_2
model_2.evaluate(test_data)
# %%
# ! what have we done till now
# 1. we trained model_2 with 10% of data with data_augmentation and saved checkpoints weights
# 2. fine-tuning model_2 with 5 more epochs and unfreezing last 10 layers with 10% data and data_augmentation (model_3)
# 3. reload model_2 using checkpoint and fine-tune model_2 with 5 more epochs and unfreezing last 10 layers with 100% data and data_augmentation (model_4)

# %%
# changing base model to trainable
base_model.trainable = True

# for last 10 only
for layer in model_2.layers[2].layers[:-10]:
    layer.trainable = False

# verifing if layers are trainable 
for layer_number , layer in enumerate(model_2.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)
# %%
# after loading, comes compiling
model_2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics=['accuracy'])

initial_epochs = 5
# fitting model_2
fine_tune_epochs = 5 + initial_epochs

history_fine_10_classes_full = model_2.fit(train_data_10_classes_full,
                                           epochs = fine_tune_epochs,
                                           steps_per_epoch = len(train_data_10_classes_full),
                                           initial_epoch = history_10_percent_data_aug.epoch[-1],
                                           validation_data = test_data,
                                           validation_steps=int( 0.25 * len(test_data)),
                                           callbacks = [create_tensorboard_callback(dir_name="transfer_learning",
                                                                                    experiment_name = "full 10 classes fine tune last 10")])
# %%
results_fine_tune_full_data = model_2.evaluate(test_data)
# %%
compare_historys(original_history=history_10_percent_data_aug,
                 new_history = history_fine_10_classes_full,
                 initial_epochs=5)
# %%
# Viewing our experiment on Tensorboard

# tensorboard dev upload --logdir ./transfer_learning \
#   --name "Transfer learning experiments" \
#   --description "A series of different transfer learning experiments with varying amounts of data and fine-tuning" \
#   --one_shot # exits the uploader when upload has finished

# how to delete experiment
# tensorboard dev delete --experiment_id 9bJ16kTYSQuxuKa3H2fLXg

# how to view list of experiment
# tensorboard dev list
# %%
# creating a function to view random image
def random_image_viewer_and_predictions(dir_name, model, class_name = "random"):
    """
    plot a random image from dir_name and pridict from model
    """
    
    if class_name == "random":
        class_name = random.choice(os.listdir(dir_name))
        class_path = dir_name + "/" + class_name 
        random_img_name = random.choice(os.listdir(class_path))
        img_path = class_path + "/" + random_img_name
        print(f"real value = {class_name}")
    else:
        class_path = dir_name + '/' + class_name
        random_img_name = random.choice(os.listdir(class_path))
        img_path = class_path + "/" + random_img_name
        print(f"real value = {class_name}")
    
    # taking image tensor
    img = mpimg.imread(img_path)
    # creating batch for this img
    img = tf.expand_dims(img, axis = 0)
    # changing shape from (512,512,3) to (224,224,3)
    img = tf.image.resize(img, size=(224,224))
    # prediction from model
    pred = model.predict(img)
    # making sense of pred
    pred = tf.argmax(pred,axis=1)[0]
    # creating pred_class
    pred_class = os.listdir(dir_name)[pred]
    print(pred_class)
    
    # removing batch
    img_for_plt = tf.squeeze(img, axis = 0)
    # chaning data from float to int
    img_for_plt = tf.cast(img_for_plt, dtype = tf.int32)
    # ploting image and naming predictions
    plt.figure(figsize=(10,7))
    plt.imshow(img_for_plt)
    plt.axis(False)
    if pred_class == class_name:
        color = 'green'
    else:
        color = 'red'
    plt.title(f"real class = {class_name} \n pred class = {pred_class}", c = color)

# %%
random_image_viewer_and_predictions(train_dir,model_2,class_name="random")

# %%
