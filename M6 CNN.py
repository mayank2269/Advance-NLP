import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout


(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
# augmentation
datagen =ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)


model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3),activation='relu'),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
history =model.fit(datagen.flow(x_train, y_train, batch_size=64),
                   epochs=50,
                   validation_data=(x_test,y_test)
                   )

loss, acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {acc}')
predictions = model.predict(x_test)
model.save('cnn_model.h5')
# # Load the model
# model = load_model('cnn_model.h5')
# # Make predictions
# predictions = model.predict(x_test)
