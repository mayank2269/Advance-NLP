# CIFAR-10  image classification using tensorflow
import tensorflow as tf
from keras import layers,models,datasets

# load data
(train_img,train_lables),(test_img,test_lables)=datasets.cifar10.load_data()
# normalise
train_img,test_img=train_img/255.0,test_img/255.0

# define model
model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10)
])

# compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train
history = model.fit(train_img, train_lables, epochs=10, 
                    validation_data=(test_img, test_lables))

# evaluation
test_loss,test_acc=model.evaluate(test_img,test_lables,verbose=2)
print(f'test acc: {test_acc}')