import numpy as np
import matplotlib.pyplot as plt 
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sig_derv(x):
    sig=sigmoid(x)
    return sig*(1-sig)

x=np.linspace(-10,10,100)
y=sigmoid(x)
dy=sig_derv(x)
plt.figure(figsize=(12,6))
plt.plot(x,y,label="sigmoid")
plt.plot(x,dy,label="derivative")
plt.xlabel("x")
plt.ylabel("value")
plt.legend()
plt.grid()
plt.show()