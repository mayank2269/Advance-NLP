# making data first 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)

plt.scatter(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.title('dataaa')
plt.show()

# add bias
x_b=np.c_[np.ones((100,1)),x]

theta = np.random.randn(2,1)
lr=0.1
epoch=1000
m=len(x_b)

# gd
for iteration in range (epoch):
    grad=2/m*x_b.T.dot(x_b.dot(theta)-y)
    theta =theta - lr*grad

print(theta)

# ploting new data
# Plot the linear regression line
plt.scatter(x, y)
plt.plot(x, x_b.dot(theta), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.show()
