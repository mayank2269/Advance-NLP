import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Generate data
x = np.linspace(-10, 10, 100)
y = relu(x)
dy = relu_derivative(x)

# Plot ReLU function and its derivative
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='ReLU')
plt.plot(x, dy, label='ReLU Derivative')
plt.title('ReLU Function and Its Derivative')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
