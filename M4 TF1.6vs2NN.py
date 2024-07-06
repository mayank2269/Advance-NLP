
# -----------------------------------TF 1.6--------------------------------------------

#  in tf 1.6 the nn was made mannually like this
import tensorflow as tf
# assumed function
def  get_batch():
    pass

# Define placeholders for input and output
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))

# Define weights and biases
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 128]))
W3 = tf.Variable(tf.random_normal([128, 10]))
b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([128]))
b3 = tf.Variable(tf.random_normal([10]))

# Define the model
layer1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))
logits = tf.add(tf.matmul(layer2, W3), b3)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Define accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize variables
init = tf.global_variables_initializer()

# Training the model
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        # Assume we have a function get_batch() that fetches batches of data
        batch_x, batch_y = get_batch()
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if epoch % 1 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            print(f'Epoch {epoch+1}, Accuracy: {acc}')


# -----------------------------------TF 2.0--------------------------------------------

# but with keras the tf2 become more user friendly and easy to use

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten

# assumed vars
train_images, train_labels=0,0
test_images, test_labels=0,0
# Define the model
model = Sequential([
    Flatten(input_shape=(784,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the model
# Assume we have training data as train_images and train_labels
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
