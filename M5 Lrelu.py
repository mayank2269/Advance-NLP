import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Generate data
x = np.linspace(-10, 10, 100)
y = leaky_relu(x)
dy = leaky_relu_derivative(x)

# Plot Leaky ReLU function and its derivative
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Leaky ReLU')
plt.plot(x, dy, label='Leaky ReLU Derivative')
plt.title('Leaky ReLU Function and Its Derivative')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
