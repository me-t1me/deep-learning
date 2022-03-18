# %%
# covering -
# tensors
# getting info from Tensors
# manupulating Tensors
# tensors - numpy
# using @tf.functions(speeding up python function)
# using GPUs
# %%
# into to tensors
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
print(tf.__version__)
# %%
# creating constant
scaler = tf.constant(7)
scaler
scaler.ndim

vector = tf.constant([10, 10])
vector
vector.ndim

matrix = tf.constant([[10, 7], [7, 10]])
matrix
matrix.ndim
an_matrix = tf.constant(
    [
        [10., 7.],
        [3., 2.],
        [8., 9.]
    ],
    dtype=tf.float16
)
an_matrix
an_matrix.ndim
# %%
tensor = tf.constant([[[1, 2, 3], [4, 5, 6]],
                      [[7, 8, 9], [10, 11, 12]],
                      [[13, 14, 15], [16, 17, 18]]
                      ])
tensor
tensor.ndim
# scaler: a single number
# vector: a number with direction
# matrix: a 2-dimensitional arry of number
# tensor: n-dimenstional array of numbers
# %%
# variable
c_vector = tf.Variable([10, 7])
vector = tf.constant([10, 7])
c_vector[0].assign(7)
c_vector
vector[0].assign(7)
# %%
# creating random tensor
ran_1 = tf.random.Generator.from_seed(7)
ran_1 = ran_1.normal(shape=(3, 2))
ran_2 = tf.random.Generator.from_seed(7)
ran_2 = ran_2.normal(shape=(3, 2))
ran_2
ran_1, ran_2, ran_1 == ran_2
# %%
# shuffling element of tensors (so inherient order donot effect learning)
n_shuffled = tf.constant([
    [10, 7],
    [3, 4],
    [2, 5],
])
tf.random.set_seed(42)  # gob?al
tf.random.shuffle(n_shuffled, seed=42)  # operation
n_shuffled
# %%
# other ways to make tensor
tf.ones([10, 7])
tf.zeros((10, 7))
# turn numpy into tensors (difference tensor can be run GPU)
num_A = np.arange(1, 25, dtype=np.int32)
A = tf.constant(num_A, shape=(3, 8))
B = tf.constant(num_A)
A, B
# %%
# getting information from tensors
# shape , rank(number of dimension), dimension(perticular dimension), size
rank_4_tensor = tf.zeros((2, 3, 4, 5))
print("Datatype:", rank_4_tensor.dtype)
print("Number of dimension:", rank_4_tensor.ndim)
print("shape:", rank_4_tensor.shape)
print('ELement along the 0 axis:', rank_4_tensor.shape[0])
print('Elements along the last axis', rank_4_tensor.shape[-1])
print('total number elements:', tf.size(rank_4_tensor).numpy())
# %%
# indexing tensor
rank_4_tensor[:2, :2, :2, :2]
# expanding tensor
rank_2_tensor = tf.constant([
    [10, 7],
    [7, 4]
])
rank_3_tensor = rank_2_tensor[:, tf.newaxis, :]
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
rank_3_tensor
# alternative way
tf.expand_dims(rank_2_tensor, axis=-1)  # means add new axis at -1 (means last)
# %%
# manupulating tensor
# + , - , * , /
tensor = tf.constant(
    [
        [10, 7],
        [3, 4]
    ]
)
tensor + 10
tensor - 10
tensor * 10
tensor / 10
# alternative
tf.multiply(tensor, 10)  # for use of GPU
# %%
# matrix multiplication
tf.matmul(tensor, tensor)  # matrix multiplication
tensor * tensor  # this for element wise multiplication python
tensor @ tensor  # matrix multi in python
x = tf.constant([
    [1, 2],
    [3, 4],
    [5, 6]
])
y = tf.constant([
    [7, 8],
    [9, 10],
    [11, 12]
])
# changing shape of y
y = tf.reshape(y, (2, 3))
tf.matmul(x, y)
# reshape x
y = tf.constant([
    [7, 8],
    [9, 10],
    [11, 12]
])
# this thing is tensor but python calculated it
tf.reshape(x, shape=(2, 3)) @ y

# this is also tensor but GPU and tf calculated it
tf.matmul(tf.reshape(x, shape=(2, 3)), y)

# transpose
x
tf.transpose(x)  # makes row into coloumn and column into rows
# take elements from next row to chance shapes
tf.reshape(x, shape=(2, 3))
tf.matmul(tf.transpose(x), y)

# tensor dotproduct

tf.tensordot(x, tf.transpose(y), axes=1) == tf.tensordot(
    x, tf.reshape(y, shape=(2, 3)), axes=1)
# %%
# changing data type
b = tf.constant([
    1.7, 7.4
])
b.dtype  # float
b = tf.constant([1, 7])
b.dtype  # int
# using cast for changing
b = tf.cast(b, dtype=tf.int16)  # faster than int32
b.dtype
# %%
# aggergating tensor
d = tf.constant([-7, -10])
tf.abs(d)
# get the minimum
# get the max
# get the mean
# sum of tensor
a = tf.constant(np.random.randint(0, 100, size=50), shape=(50))
print(a)
print(tf.reduce_mean(a).numpy())
print(tf.reduce_max(a).numpy())
print(tf.reduce_min(a).numpy())
print(tf.reduce_sum(a).numpy())
# for variance and stds install tfp
# from tensor flow
print(tf.math.reduce_variance(tf.cast(a,  dtype=tf.float32)).numpy())
print(tfp.stats.variance(a, sample_axis=0).numpy())  # from tfp
# we have to use math and change dtype to float
print(tf.math.reduce_std(tf.cast(a,  dtype=tf.float32)).numpy())
# %%
# positional max and min
tf.random.set_seed(42)
f = tf.random.uniform(shape=(50,))
tf.argmax(f)  # index of largest number
tf.reduce_max(f) == f[tf.argmax(f)]
tf.argmin(f)
tf.reduce_min(f) == f[tf.argmin(f)]
# %%
# removing all single dimensions
tf.random.set_seed(42)
l = tf.constant(tf.random.uniform(shape=[50]), shape=[1, 1, 1, 1, 50])
l
l_se = tf.squeeze(l)  # remove dimension with 1 as shape
l_se
# %%
# one hot encoding tensor
# this for features
list_A = [0, 1, 2, 3]
tf.one_hot(list_A, depth=5)
# %%
# squaring, log, sqaure root
a = tf.range(1, 10)
print(tf.square(a))
print(tf.sqrt(tf.cast(x,  dtype=tf.float32)))  # square root
print(tf.math.log(tf.cast(x,  dtype=tf.float32)))  # log
# %%
# using numpy and tensor (single tensor is also a tensor but single numpy array is a number which can be used in operations)
j = tf.constant(np.array([3., 7., 10.]))  # converting numpy array into tensor

j.numpy(), np.array(j), type(np.array(j)), type(
    j.numpy())  # 2 ways to convert tensor into numpy

j_with_list = tf.constant([3., 7., 10.])
# here j is float64 and j_with_list is float33
print(j.dtype, j_with_list.dtype)

# %%
