# %% Problem 1
import numpy as np
from matplotlib import pyplot as plt
n = 50
x = np.linspace(0, 1, n)
a = 5; b = -4; c = 3
y = a*x*x + b*x + c + 0.5*np.random.randn(n)
# %% Plot data point
fig1 = plt.figure(figsize=(7,5), dpi = 300)
ax1 = fig1.add_axes([0.1,0.1,1,1])
ax1.plot(x,y,'o')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Data Point')
fig1.savefig('data_point.png', dpi = 300, bbox_inches = 'tight')
fig1.show()

# %% Problem 4
a11 = np.sum(x**4)
a12 = np.sum(x**3)
a13 = np.sum(x**2)
a23 = np.sum(x)
a33 = np.sum(n)
A = np.array([[a11, a12, a13], [a12, a13, a23], [a13, a23, a33]])
b_dir = np.array([np.sum(x*x*y), np.sum(x*y), np.sum(y)])
m0 = np.linalg.inv(A).dot(b_dir)
print(m0)

# %% Problem 6
# Create gradient function
def grad(m, x, y):
    d = m[0]*x*x + m[1]*x +m[2] - y
    return np.array([np.sum(d*x*x), np.sum(d*x), np.sum(d)])

def grad_sgd(m, x, y, i):
    d = m[0]*x[i]*x[i] + m[1]*x[i] +m[2] - y[i]
    return np.array([np.sum(d*x[i]*x[i]), np.sum(d*x[i]), np.sum(d)])

# Create step function
def step(m, x, y, g):
    d = m[0]*x*x + m[1]*x +m[2] - y
    dd = g[0]*x*x + g[1]*x + g[2]
    return np.sum(d*dd)/np.sum(dd*dd)

# Create loss function
def loss(m, x, y):
    e = m[0]*x*x + m[1]*x +m[2] - y
    return 0.5*np.sum(e*e)
# %% Implement batch gradient descent method (BGD)
m = np.random.randn(3)
l = {}
l[0] = loss(m,x,y)
i = 0
while i <= 5000:
    g = grad(m, x, y)
    s = step(m, x, y, g)
    m = m - s*g
    i += 1
    l[i] = loss(m, x, y)
    print(l[i])
    if l[i-1] - l[i] <= 0.0001:
        break
# %% Implement Stochastic gradient descent method (SGD)
lr = 0.0005
ms = np.random.randn(3)
lsgd = {}
lsgd[0] = loss(ms, x, y)
i = 0
while i <= 200000:
    for j in np.random.permutation(n):
        gs = grad_sgd(ms, x, y, j)
        ms = ms - lr*gs
    i += 1
    lsgd[i] = loss(ms,x,y)
    print(lsgd[i])
    if lsgd[i-1] - lsgd[i] <= 0.00001:
        break
# %% Plot loss of BGD
fig2 = plt.figure(figsize=(7,5), dpi=300)
ax2 = fig2.add_axes([0.1, 0.1, 1, 1])
ax2.plot(l.keys(),l.values())
ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.set_title('Loss Values using Batch Gradient Descent Method')
fig2.show()
fig2.savefig('bgd.png', dpi = 300, bbox_inches = 'tight')
print(m)
# %% Plot loss of SGD
fig3 = plt.figure(figsize=(7,5), dpi = 300)
ax3 = fig3.add_axes([0.1, 0.1, 1, 1])
ax3.plot(lsgd.keys(),lsgd.values())
ax3.set_xlabel('X')
ax3.set_ylabel('y')
ax3.set_title('Loss Values using Stochatic Gradient Descent Method')
fig3.show()
fig3.savefig('sgd.png', dpi = 300, bbox_inches = 'tight')
print(ms)
# %% Plot data with fit line
xplot = np.linspace(x.min(), x.max(), 100)
yplot = m0[0]*xplot*xplot + m0[1]*xplot +m0[2]
yplot2 = m[0]*xplot*xplot + m[1]*xplot +m[2]
yplot3 = ms[0]*xplot*xplot + ms[1]*xplot +ms[2]
fig4 = plt.figure(figsize=(7,5), dpi = 300)
ax4 = fig4.add_axes([0.1,0.1,1,1])
ax4.plot(x,y,'o')
ax4.plot(xplot, yplot, label='Direct methods')
ax4.plot(xplot, yplot2, label='Batch Gradient Descent method')
ax4.plot(xplot, yplot3, label='Stochastic Gradient Descent method')
ax4.set_xlabel('X')
ax4.set_ylabel('y')
ax4.set_title('Data Point with Method Line')
ax4.legend()
fig4.savefig('data_line_dir.png', dpi = 300, bbox_inches = 'tight')
fig4.show()
# %%
