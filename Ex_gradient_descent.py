# %%
import numpy as np
import matplotlib.pyplot as plt
#%%
for i in np.arange(0,10):
    print(i)
# %%
n = 50
x = np.linspace(0, 1, n)
a = 5; b = -4; c = 3
y = a*x*x + b*x + c + 0.5*np.random.randn(n)
plt.plot(x,y,'o')
plt.show()

# m = [w,b]
def grad(m,x,y):
    e = m[0]*x+m[1]-y
    return np.array([np.sum(e*x), np.sum(e)])

# g = [gw,gb]
def step(m,x,y,g):
    e = m[0]*x+m[1]-y
    gg = g[0]*x+g[1]
    return np.sum(e*gg)/np.sum(gg*gg)

# loss function
def loss(m,x,y):
    e = m[0]*x+m[1]-y
    return 0.5*np.sum(e*e)

def sgd(m, x, y, i):
    g = m[0]*x[i] + m[1] - y[i]
    return np.array([g*x[i], g])

# %% SGD
LR = 0.1
# alpha = 0.01
n_epoch =100
m = np.random.randn(2)
l = np.zeros(n_epoch+1)
l[0] = loss(m,x,y)
print(loss(m, x, y))

for i in range(n_epoch):
    for j in np.random.permutation(n):
        g = sgd(m, x, y, j)
        m = m - LR*g
    l[i+1] = loss(m, x, y)

plt.plot(np.arange(n_epoch+1),l)


# %% Minibatch
n  = 50
b = 10
nb = int(n/b)
i = np.random.permutation(n).reshape(np,b)
for i in range(nb):
    xb, yb, = x[i]
