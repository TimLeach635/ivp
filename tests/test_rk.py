from one_step import generate_runge_kutta
import matplotlib.pyplot as plt
import numpy as np

# test RK4, a famous method
a = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
b = np.array([[1/6], [1/3], [1/3], [1/6]])
rk4 = generate_runge_kutta(a, b)
hs = list(map(lambda x: 2**x, range(0, -10, -1)))
max_errors = []

p = lambda x: -1
q = lambda x: 0

for h in hs:
    xs, ys = rk4(0, 1, h, x_max=2, p=p, q=q)
    max_errors.append(np.max(np.abs(np.exp(xs) - ys)))

# plot error
plt.loglog(hs, max_errors, '.')
plt.xlabel(r'$h$')
plt.ylabel('Maximum error')
plt.title('Errors')
plt.show()

# find slope
slope, _ = np.polyfit(np.log(hs), np.log(max_errors), 1)
print('Slope =', slope)
