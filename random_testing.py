import tensorflow as tf
from tensorflow import keras


w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1.,2.,3.]]


with tf.GradientTape() as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)

model = keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)