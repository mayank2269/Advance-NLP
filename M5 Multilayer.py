import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
import numpy as np

input = np.array([[1,1],[1,0],[0,1],[0,0]],dtype=float)
output = np.array([[1],[0],[0],[0]],dtype=float)

model=Sequential()
model.add(Dense(3,input_dim=2,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(input,output,epochs=50)

loss,acc=model.evaluate(input,output)

print(acc)