import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def linear_approximation(x):
    a = []
    for item in x:
        a.append((1/(1+math.exp(-1))-0.5)*item + 0.5)
    return a


x = np.arange(-10., 10., 0.2)
x_ = np.arange(-2.5, 2.5, 0.2)

sig = sigmoid(x)
approx_sig = linear_approximation(x_)
plt.figure(figsize=(10,7))
plt.axvline(1, c='r')
plt.axvline(-1, c='r')
plt.plot(x_,approx_sig, '--', color='orange')
plt.plot(x,sig, color='orange')
plt.title("Linear region of sigmoid function")
plt.savefig('imgs/sigmoid-linear-region.png', dpi=100)
plt.show()