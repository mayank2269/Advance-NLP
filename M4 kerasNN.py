# libs 
import tensorflow as tf
from keras import layers,models

# define model
# i/p layer
inputs = tf.keras.Input(shape=(784,))
# hidden layer
x=layers.Dense(512, activation='relu')(inputs)
x=layers.Dropout(0.2)(x)
x=layers.Dense(256, activation='relu')(x)
x=layers.Dropout(0.2)(x)

# o/p layer
outputs =layers.Dense(10,activation='softmax')(x)
# make model
model = models.Model(inputs=inputs, outputs=outputs)

# compilr it
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# lets use the dataset MINST and preprocces and load it
mnist =tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalise it (optional)
x_train,x_test =x_train/255.0,x_test/255.0
# Flatten images to 1D vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# training of nn
model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_test,y_test))

# check accuracyy
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy}')
