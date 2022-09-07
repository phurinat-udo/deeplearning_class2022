# %%
import numpy as np
import seaborn as sns
# %%
n = 50
x = np.linspace(0, 1, n)
w = 5; b = -4
y = w*x + b + np.random.randn(n)
sns.scatterplot(x, y)
# %%
A11 = np.sum(x*x)
A12 = np.sum(x)
A = np.array([[A11, A12], [A12, n]])
b_dir = np.array([np.sum(x*y), np.sum(y)])
m = np.linalg.inv(A).dot(b_dir)
# %%
y_dir = m[0]*x+m[1]
sns.scatterplot(x, y)
sns.lineplot(x, y_dir)
# %%
# def grad():