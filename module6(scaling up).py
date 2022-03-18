# %%
# import helper functions
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import wget
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir, make_confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import itertools
import pandas as pd
import random
import os
# %%
# downloading dataset # ! one time
wget.download("https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip")
unzip_data("101_food_classes_10_percent.zip")
# %%
train_dir = "101_food_classes_10_percent/train"
test_dir = "101_food_classes_10_percent/test"

# how many classes/photos are there
walk_through_dir(test_dir)
# %%
# data imports
IMG_SIZE = (224,224)
train_data_all_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                label_mode = 'categorical',
                                                                                image_size=IMG_SIZE)
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode='categorical',
                                                                image_size=IMG_SIZE,
                                                                shuffle=False) # don't shuffle test dataset

# %%
# TODO what am I going to do
# create ModelCheackpoint callback
# create data_augmented_layer
# build a headless model (meaning no top layers) of EfficientnetB0
# compile the model
# run it for 5 epochs (validate on 15% data)
# %%
# creating ModelCheckpoint callback
checkpoint_path = "101_classes_10_percent_data_model_checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         monitor='val_accuracy')
# %%
# creating data augmentation layer
data_augmentation = Sequential([
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
], name="data_augmentation")
# %%
# creating base_model
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# creating structure of model
inputs = layers.Input(shape = (224,224,3), name="input_layer")
x = data_augmentation(inputs)
x = base_model(x, training = False) # ! so that weights stay frozen
x = layers.GlobalAveragePooling2D(name = "global_average_pooling_2D")(x)
outputs = layers.Dense(len(train_data_all_10_percent.class_names) , activation="softmax", name = "output_layer")(x)

# creating model
model = tf.keras.Model(inputs = inputs, outputs = outputs)
# %%
# compiling model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
# %%
# fitting model
# ! don't use multiprocess in fit method it slows down process
initial_epochs = 5
history = model.fit(train_data_all_10_percent,
                    epochs=initial_epochs,
                    steps_per_epoch=len(train_data_all_10_percent),
                    validation_data=test_data,
                    validation_steps=int(0.15 * len(test_data)),
                    callbacks=[checkpoint_callback])
# %%
feature_extraction_results = model.evaluate(test_data)
feature_extraction_results
plot_loss_curves(history) # ! graph suggest overfitting (fine-tuning will work)
# %%
model.load_weights(checkpoint_path)
# unfreezing all layers
base_model.trainable = True
# refreezing 95% of them
for layer in base_model.layers[:-5]:
    layer.trainable = False
# verifing result
for layer_number, layer in enumerate(base_model.layers):
    print(layer_number, layer.name, layer.trainable)
# %%
# recompile 
# ! decrease learning rate by 10x
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
              metrics=['accuracy'])
# %%
# fitting model
history_all_classes_10_percent_fine_tune = model.fit(train_data_all_10_percent,
                              epochs = 10,
                              steps_per_epoch=len(train_data_all_10_percent),
                              validation_data=test_data,
                              validation_steps=int(0.15*len(test_data)),
                              initial_epoch=history.epoch[-1])
# %%
# evalute model
all_classes_10_percent_fine_tune = model.evaluate(test_data)
all_classes_10_percent_fine_tune
# %%
# compare the histories
compare_historys(history,history_all_classes_10_percent_fine_tune,initial_epochs = 5) # ! this shows fine tuning model is overfitting
# %%
# saving model
model.save('101_food_classes_10_percent_saved_big_model')
# %%
# load nd evaluate saved model
loaded_model = tf.keras.models.load_model('101_food_classes_10_percent_saved_big_model')
# %%
loaded_model_results = loaded_model.evaluate(test_data)
loaded_model_results
# %%
all_classes_10_percent_fine_tune
# %% # ! no need to use it now
wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/06_101_food_class_10_percent_saved_big_dog_model.zip')
unzip_data('06_101_food_class_10_percent_saved_big_dog_model.zip')
# %%
# load model in saved model
model = tf.keras.models.load_model('06_101_food_class_10_percent_saved_big_dog_model')
# %%
# evaluate our model
results_downloaded_model = model.evaluate(test_data)
# %%
pred_prob = model.predict(test_data, verbose = 1)
# %%
pred_prob.shape  # ? shape = (25250, 101)
test_data.class_names[tf.argmax(pred_prob[0])]
pred_classes = pred_prob.argmax(axis = 1)
# %%
y_batch = []
for image, label in test_data.unbatch():
    y_batch.append(label.numpy().argmax())
# %%
y_batch[:10]
# %%
# cheaking accuracy using sklearn
sklearn_accuracy = accuracy_score(y_batch, pred_classes)
sklearn_accuracy
# %%
# ? does this metric come close to the model.evaluate
np.isclose(sklearn_accuracy, results_downloaded_model[1])
# %% # ! we have to make xaxis lable visible
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()
  
  # ? chaning xlabel rotation
  plt.xticks(rotation = 70, fontsize = text_size)
  plt.yticks(fontsize = text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
# %%
class_names = test_data.class_names
class_names
# %%
# ? lets visualize using confusion matrix
make_confusion_matrix(y_true=y_batch,
                      y_pred=pred_classes,
                      classes=class_names,
                      figsize=(100,100),
                      text_size=20,
                      savefig = True)
# %%
# ? lets create a classification report
classification_report_dict = classification_report(y_true=y_batch, y_pred=pred_classes, output_dict=True)
classification_report_dict
# %%
# ploting all F1 scores
classification_report_dict
# %%
class_f1_scores = {}
# ? one element of dict is called items and dict.item() can be used to iterate
# ? item is tuple of key and value, dict.keys() and dict.values() can be used to iterate
for k, v in classification_report_dict.items():
  if k == 'accuracy':
    break
  else:
    class_f1_scores[class_names[int(k)]] = v['f1-score']
  
class_f1_scores
# %%
# converting directory into dataframe
f1_scores = pd.DataFrame({"class_names": list(class_f1_scores.keys()),
                          "f1-score": list(class_f1_scores.values())})
# sorting DataFrame
f1_scores = f1_scores.sort_values('f1-score', ascending=False)
# %%
# plot this DataFrame
f1_scores
# %%
# creating a autolabel function to show f1-scores at right side of bar
def autolabel(rects):
  for rect in rects:
    width = rect.get_width()
    ax.text(1.03*width, rect.get_y() + rect.get_height()/1,
            f"{width:.2f}",
            ha = 'center', va = 'bottom')
# %%
fig, ax = plt.subplots(figsize=(12,25))
scores = ax.barh(range(len(f1_scores)), f1_scores['f1-score'].values)
ax.set_yticks(range(len(f1_scores)))
ax.set_yticklabels(f1_scores['class_names'])
ax.set_xlabel("f1-score")
ax.set_title("F1 score for 101 food classes")
ax.invert_yaxis()
autolabel(scores)
plt.show()

# %%
# ! how to visualize test images
# ? read the image with tf.io.read()
# ? convert into tensor with tf.io.decode_image()
# ? resize images with tf.image.resize()

def load_and_prep_image(filename, image_size = 224, scale = True):
  """
  read the file from filename and convert it into tensor, and resize it into (image_size, image_size, 3)
  """
  image = tf.io.read_file(filename)
  image = tf.io.decode_image(image, channels = 3)
  image = tf.image.resize(image, [image_size, image_size])
  if scale:
    return image/255.
  else:
    return image
# %%
# ? use random images, predict on them and plot these images with prediction and prediction probabilities

def visualizer(dir=test_dir, class_names=class_names):
  test_dir = "101_food_classes_10_percent/test"
  random_class = random.choice(os.listdir(test_dir))
  class_path = test_dir + "/" + random_class
  random_img = random.choice(os.listdir(class_path))
  random_img_path = class_path + "/" + random_img
  img = load_and_prep_image(random_img_path, scale = False)
  img = tf.expand_dims(img, axis = 0)
  prediction = model.predict(img)
  pred = tf.argmax(prediction, axis = 1)
  pred_prob = tf.reduce_max(prediction, axis = 1).numpy()[0]
  str_pred = class_names[pred.numpy()[0]]

  # ploting images 
  image = mpimg.imread(random_img_path)
  if str_pred == random_class:
    title_color = 'g'
  else:
    title_color = 'r'
  plt.imshow(image);
  plt.title(f"actual class: {random_class}, pred: {str_pred}, prob: {pred_prob:.2f}", color=title_color)
  plt.axis('off')

# %%
plt.figure(figsize=(17,10))
for i in range(3):
  plt.subplot(3,1,i+1)
  visualizer()

# %%
# ! finding most wrong predictions -> 1. wrong prediction 2. high percentage

# 1. listing files
file_paths = []
for file_path in test_data.list_files("101_food_classes_10_percent/test/*/*.jpg", shuffle=False):
  file_paths.append(file_path.numpy())
# %%
# 2. create a dataframe for all parameters such as prediction percentage
pred_df = pd.DataFrame({"img_path": file_paths,
                        "y_true": y_batch,
                        "y_pred": pred_classes,
                        "pred_conf": pred_prob.max(axis=1),
                        "y_true_classname": [class_names[i] for i in y_batch],
                        "y_pred_classname": [class_names[i] for i in pred_classes]})
# %%
# 3. using our df to find wrong predictions
pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
# %%
# 4. sorting df for high wrong prediction values
top_100_wrong = pred_df[pred_df["pred_correct"] == False].sort_values(by="pred_conf", ascending = False, ignore_index = True)[:100]
# %%
# 5. ploting these test data
plt.figure(figsize=(15,10))
inde = 20
for i in top_100_wrong.index[inde:inde+9]:
  plt.subplot(3,3,i+1-inde)
  img_path = top_100_wrong["img_path"][i]
  img = mpimg.imread(img_path.decode("utf-8"))
  plt.imshow(img)
  plt.axis("off")
  plt.title(f"real class: {top_100_wrong['y_true_classname'][i]}, pred class: {top_100_wrong['y_pred_classname'][i]} \nconfidence: {top_100_wrong['pred_conf'][i]}")

# %%
# getting costum image
wget.download("https://storage.googleapis.com/ztm_tf_course/food_vision/custom_food_images.zip")

unzip_data("custom_food_images.zip")
# %%
file_paths = [ "custom_food_images/" + img_path for img_path in os.listdir("custom_food_images")]
file_paths
# %%
plt.figure(figsize=(10,10))
for i,img in enumerate(file_paths):
  plt.subplot(3,2,i+1)
  img = load_and_prep_image(img, scale = False)
  img = tf.expand_dims(img, axis=0)
  pred_prob = model.predict(img)
  pred_conf = pred_prob.max(axis=1)[0]
  pred_class = pred_prob.argmax(axis=1)[0]
  pred_classname = class_names[pred_class]
  img = tf.squeeze(img, axis=0)
  img = img/255.
  plt.imshow(img)
  plt.axis(False)
  plt.title(f"pred_class: {pred_classname} \nprob: {pred_conf:.2f}")
  
# %%
