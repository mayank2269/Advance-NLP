import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Generate data
x = np.linspace(-10, 10, 100)
y = tanh(x)
dy = tanh_derivative(x)

# Plot tanh function and its derivative
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='tanh')
plt.plot(x, dy, label='tanh Derivative')
plt.title('Tanh Function and Its Derivative')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
