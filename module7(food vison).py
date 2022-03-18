# %%

import tensorflow_datasets as tfds
from helper_functions import (
    create_tensorboard_callback,
    plot_loss_curves,
    compare_historys,
)
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import pickle
import matplotlib.pyplot as plt


# sklearn, matplotlib

# %%
(train_data, test_data), ds_info = tfds.load(
    name="food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,
    data_dir="D:/coding project/tensorflow_datasets",
    with_info=True,
)

# %%
class_name = ds_info.features["label"].names
class_name[:10]
# %%
train_one_sample = train_data.take(1)

for image, label in train_one_sample:
    print(f"""
          Image shape: {image.shape}
          Image data_type: {image.dtype}
          tensor form: {label}
          Class name: {class_name[label.numpy()]}
          """)
    plt.imshow(image)
    plt.title(class_name[label.numpy()])
    plt.axis(False)
# %%
# ! creating preprocessing function
# ? what we know -
# uint8
# size of pic is different
# not scaled
# ? what we need -
# ! float32
# ! resizing to 224
# ! needs batches of 32
# scaled (not needed efficientnet)
# %%


def preprocess_img(image, label, img_shape=224):
    """
    convert image datatype to float32
    reshape image to [img_shape, img_shape, color channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape])
    return tf.cast(image, tf.float32), label


# %%
process_img = preprocess_img(image, label)[0]
# %%
# ! batch and prepare dataset
train_data = train_data.map(map_func=preprocess_img,
                            num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1000).batch(
    batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

test_data = test_data.map(map_func=preprocess_img,
                          num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# %%
train_data, test_data
# %%
# creating callbacks
checkpoint_path = "model_checkpoints/cp.cpkt"
model_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, moniter='val_acc', save_best_only=True, save_weights_only=True,)
# %%
# ! implementing mixed_precision
mixed_precision.set_global_policy("mixed_float16")
# %%
# ? creating base model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# create functional model
inputs = layers.Input(shape=input_shape, name="input_layer")

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(len(class_name))(x)
outputs = layers.Activation(
    "softmax", dtype=tf.float32, name="softmax_float32")(x)
model = tf.keras.Model(inputs, outputs)

# compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
# %%
model.summary()
# %%
for layer in model.layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

# %%
for layer in model.layers[1].layers[:20]:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
# %%
# ! fitting the model
history_101_food_classes_feature_extract = model.fit(train_data,
                                                     epochs=3,
                                                     steps_per_epoch=len(
                                                         train_data),
                                                     validation_data=test_data,
                                                     validation_steps=int(
                                                         0.15 * len(test_data)),
                                                     callbacks=[create_tensorboard_callback("training_logs",
                                                                                            "efficientnetb0_101_classes_all_data_feature_extract"),
                                                                model_checkpoints])
# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
history_feature_extract_model = model.evaluate(test_data)
# %%
