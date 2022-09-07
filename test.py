#%%
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
# %%
X = np.float32(np.linspace(-10, 10, 100))
X
# %%
y_sigm = []
for i in X:
    y = 1/(1+np.exp(-i))
    y_sigm.append(y)
#%%
print(y_sigm)
# %%
y_hyper = []
for i in X:
    y = np.tanh(i)
    y_hyper.append(y)
y_hyper

#%%
y_relu = []
for i in X:
    y = np.maximum(0,i)
    y_relu.append(y)

# %%
y_softplus = []
for i in X:
    y = np.log(1+np.exp(i))
    y_softplus.append(y)
print(y_softplus)
# %%
Fig = plt.figure()
plt.plot(X, y_sigm, label="Sigmoid")
plt.plot(X, y_hyper, label = "Hyperbolic tangent")
plt.plot(X, y_relu, label = "Relu")
plt.plot(X, y_softplus, label = "Softplus")
plt.legend()
plt.show()

# %%
