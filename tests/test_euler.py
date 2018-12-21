from one_step import explicit_euler, implicit_euler
import matplotlib.pyplot as plt
import numpy as np

# define grids
grids = [(10, 0.2), (20, 0.1), (40, 0.05), (80, 0.025)]

# define titles
titles = ['Explicit Euler, linear version',
          'Explicit Euler, nonlinear version',
          'Implicit Euler, linear version',
          'Implicit Euler, nonlinear version']

# define linear problem
p = lambda x: -1
q = lambda x: 0

# define nonlinear problem
f = lambda x, y: y
f_prime = lambda x, y: 1

# define methods
methods = [lambda h, n: explicit_euler(0, 1, h, kind='linear', p=p, q=q, n=n),
           lambda h, n: explicit_euler(0, 1, h, kind='nonlinear', f=f, n=n),
           lambda h, n: implicit_euler(0, 1, h, 1, kind='linear', p=p, q=q, n_pts=n),
           lambda h, n: implicit_euler(0, 1, h, 1, kind='nonlinear', f=f, f_prime=f_prime, n_pts=n)]

# solve
for i in range(4):
    for n, h in grids:
        # solve for given grid
        xs, ys = methods[i](h, n)

        # plot numerical solution
        plt.subplot(2, 4, i+1)
        plt.plot(xs, ys, label=r'$h = {}$'.format(h))

        # plot error
        plt.subplot(2, 4, i+5)
        plt.plot(xs, np.abs(np.array(ys) - np.exp(xs)), label=r'$h = {}$'.format(h))
    plt.legend()
    plt.title('Error')
    plt.subplot(2, 4, i+1)
    # this next line is giving me an error, but I know xs will always be defined by this point
    plt.plot(xs, np.exp(xs), label=r'Exact $e^x$')
    plt.legend()
    plt.title(titles[i])
plt.show()
