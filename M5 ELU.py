import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x >= 0, 1, alpha * np.exp(x))

# Generate data
x = np.linspace(-5, 5, 100)
y = elu(x)
dy = elu_derivative(x)

# Plot ELU function and its derivative
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='ELU')
plt.plot(x, dy, label='ELU Derivative')
plt.title('ELU Function and Its Derivative')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
