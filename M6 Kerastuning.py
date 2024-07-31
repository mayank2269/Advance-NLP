import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Sequential

(xtrain,ytrain),(xtest,ytest)=cifar10.load_data()
xtrain=xtrain.astype('float32')/255.0
xtest=xtest.astype('float32')/255.0
ytrain=keras.utils.to_categorical(ytrain,10)
ytest=keras.utils.to_categorical(ytest,10)

# coonvolution layer 1
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

# coonvolution layer 2
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

# coonvolution layer 3
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))\

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(xtrain,ytrain,epochs=10,batch_size=64,validation_split=0.2)
loss, accuracy = model.evaluate(xtest, ytest)
print(f'Test accuracy: {accuracy}')