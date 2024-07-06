import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

# Create a constant tensor
a = tf.constant([1, 2, 3, 4], dtype=tf.float32)
print(a)

# Create a variable tensor
b = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
print(b)
x = tf.constant(3.0)

# gradient tape
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
dy_dx = tape.gradient(y, x)
print(dy_dx)  # Output: 6.0

# dataset
# Create a dataset from a NumPy array
import numpy as np
data = np.random.randn(1000, 784)
labels = np.random.randint(10, size=1000)
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(32)
# Iterate through the dataset
for batch_data, batch_labels in dataset:
    print(batch_data.shape, batch_labels.shape)

# function
@tf.function
def my_function(x, y):
    return tf.reduce_mean(x + y)
# Use the compiled function
result = my_function(tf.constant([1, 2, 3]), tf.constant([4, 5, 6]))
print(result)  # Output: 5.0

# distribution stretegy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])



# -------------keras wale----------------


# Define a simple sequential model
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Add layers to a sequential model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
model = MyModel()

# optimiser
from keras.optimizers import Adam
# Compile the model
model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# losses
from keras.losses import SparseCategoricalCrossentropy
# Define a loss function
loss = SparseCategoricalCrossentropy()

# metrics
from keras.metrics import SparseCategoricalAccuracy
# Define a metric
metric = SparseCategoricalAccuracy()







