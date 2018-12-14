import numpy as np


def explicit_euler(f, x_0, y_0, h, n):
    # list of x-values
    xs = list(np.linspace(x_0, x_0 + n*h, n))

    # list of corresponding y-values
    ys = [y_0]

    # iterate through
    for i in range(n-1):
        ys.append(ys[i] + h*f(xs[i], ys[i]))

    return xs, ys
