import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
# example of logic gates
input=np.array([[1,1],[1,0],[0,1],[0,0]],dtype=float)
output=np.array([[1],[1],[1],[0]],dtype=float)
# define model
model = Sequential()
model.add(Dense(1,input_dim=2,activation='sigmoid'))
# compile it
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(input,output,epochs=50)
# evaluate
loss,accuracy=model.evaluate(input,output)
print(accuracy)



# prediction 

predictions = model.predict(input)
print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Input: {input[i]}, Prediction: {prediction[0]:.2f}")

