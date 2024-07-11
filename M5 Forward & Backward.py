import numpy as np

def sigmoid(x):
    return 1/(1+ np.exp(-x))

x=np.array([[1,0,1],[0,1,1],[1,1,1]])
y=np.array([[1],[0],[1]])

# wt and bias

w1=np.random.rand(3,4)
b1=np.random.rand(1,4)
w2=np.random.rand(4,1)
b2=np.random.rand(1,1)
# f/w
Z1=np.dot(x,w1)+b1
A1=sigmoid(Z1)
Z2=np.dot(A1,w2)+b2
A2=sigmoid(Z2)

print(f'ap={A2}')

def mse_loss(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)
def sig_deriv(x):
    return x*(1-x)

# leaerning rate
lr=0.1
loss=mse_loss(y,A2)
print(loss)

# b/w
da2=-(y-A2)
dz2=da2*sig_deriv(A2)
dw2=np.dot(A1.T,dz2)
db2=np.sum(dz2,axis=0,keepdims=True)

da1=np.dot(dz2,w2.T)
dz1=da1*sig_deriv(A1)
dw1=np.dot(x.T,dz1)
db1=np.sum(dz1,axis=0,keepdims=True)

# update 
w2-=lr * dw2
b2-=lr*db2
w1-=lr*dw1
b1-=lr*db1

# f/w after updation
Z1=np.dot(x,w1)+b1
A1=sigmoid(Z1)
Z2=np.dot(A1,w2)+b2
A2=sigmoid(Z2)

loss=mse_loss(y,A2)
print(loss)