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


def implicit_euler(f, f_prime, x_0, y_0, h, n):
    # list of x-values
    xs = list(np.linspace(x_0, x_0 + n*h, n))

    # list of corresponding y-values
    ys = [y_0]

    # iterate through
    for i in range(n-1):
        # do two steps of Newton root-finding iteration to find the next slope
        # create a function to find zeroes
        big_f = lambda y_m: y_m - ys[i] - h*f(xs[i+1], y_m)
        big_f_prime = lambda y_m: 1 - h*f_prime(xs[i+1], y_m)
        ys.append(root_find(big_f, big_f_prime, ys[i], 1))

    return xs, ys


def root_find(f, f_prime, init_guess, steps):
    x = init_guess
    for _ in range(steps):
        x = x - f(x)/f_prime(x)

    return x
