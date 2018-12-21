import numpy as np


def explicit_euler(x_0, y_0, h, kind='nonlinear', **kwargs):
    """Perform the explicit Euler method to numerically solve the given first-order ODE.

    Positional arguments:
        x_0   - the x-value of the initial point.
        y_0   - the y-value of the initial point.
        h     - the spacing between points.

    Keyword arguments:
        kind  - either 'linear' or 'nonlinear'.
                'linear' means the ODE is of the form dy/dx + p(x)*y = q(x), and you must pass arguments 'p' and 'q'.
                'nonlinear' means the ODE is of the more general form dy/dx = f(x, y), and you must pass 'f'.
        p     - a function that takes one numerical argument and returns a numerical value. Only relevant if kind is
                linear.
        q     - same as above.
        f     - a function that takes two numerical arguments and returns one numerical value. Only relevant if kind is
                nonlinear.
        n     - the number of points. Either define this or x_max.
        x_max - the maximum x-value, inclusive. Either define this or n.
    """
    # check enough arguments given
    if kind not in ('linear', 'nonlinear'):
        raise ValueError("'kind' must either take the value 'linear' or 'nonlinear'.")

    if kind == 'linear' and ('p' not in kwargs.keys() or 'q' not in kwargs.keys()):
        raise TypeError("Kind is linear, but 'p' and 'q' have not both been given.")

    if kind == 'nonlinear' and 'f' not in kwargs.keys():
        raise TypeError("Kind is nonlinear, but 'f' has not been given.")

    if 'n' not in kwargs.keys() and 'x_max' not in kwargs.keys():
        raise TypeError("Must define either 'n' or 'x_max'.")

    # set variables to use
    p, q, f = None, None, None
    if kind == 'linear':
        p = kwargs['p']
        q = kwargs['q']
    else:
        f = kwargs['f']

    if 'n' in kwargs.keys():
        n = kwargs['n']
    else:
        # use n in calculation both times
        n = int(kwargs['x_max']/h)

    # list of x-values
    xs = list(np.linspace(x_0, x_0 + n*h, n))

    # list of corresponding y-values
    ys = [y_0]

    # iterate through
    if kind == 'linear':
        for i in range(n-1):
            ys.append(ys[i] + h*(q(xs[i]) - p(xs[i])*ys[i]))
    else:
        for i in range(n-1):
            ys.append(ys[i] + h*f(xs[i], ys[i]))

    return xs, ys


def implicit_euler(x_0, y_0, h, n_iter, kind='nonlinear', **kwargs):
    """Perform the implicit Euler method to numerically solve the given first-order ODE.

    Positional arguments:
        x_0    - the x-value of the initial point.
        y_0    - the y-value of the initial point.
        h      - the spacing between points.
        n_iter - the number of iterations in the root-finding algorithm.
                 this program currently uses the Newton-Raphson method to find the roots, with the view to add more
                 options later.

    Keyword arguments:
        kind    - either 'linear' or 'nonlinear'.
                  'linear' means the ODE is of the form dy/dx + p(x)*y = q(x), and you must pass arguments 'p' and 'q'.
                  'nonlinear' means the ODE is of the more general form dy/dx = f(x, y), and you must pass 'f'.
        p       - a function that takes one numerical argument and returns a numerical value. Only relevant if kind is
                  linear.
        q       - same as above.
        f       - a function that takes two numerical arguments and returns one numerical value. Only relevant if kind
                  is nonlinear.
        f_prime - the derivative of f above.
        n_pts   - the number of points. Either define this or x_max.
        x_max   - the maximum x-value, inclusive. Either define this or n.
    """
    # check enough arguments given
    if kind not in ('linear', 'nonlinear'):
        raise ValueError("'kind' must either take the value 'linear' or 'nonlinear'.")

    if kind == 'linear' and ('p' not in kwargs.keys() or 'q' not in kwargs.keys()):
        raise TypeError("Kind is linear, but 'p' and 'q' have not both been given.")

    if kind == 'nonlinear' and ('f' not in kwargs.keys() or 'f_prime' not in kwargs.keys()):
        raise TypeError("Kind is nonlinear, but 'f' and 'f_prime' have not both been given.")

    if 'n_pts' not in kwargs.keys() and 'x_max' not in kwargs.keys():
        raise TypeError("Must define either 'n_pts' or 'x_max'.")

    # set variables to use
    p, q, f, f_prime = None, None, None, None
    if kind == 'linear':
        p = kwargs['p']
        q = kwargs['q']
    else:
        f = kwargs['f']
        f_prime = kwargs['f_prime']

    if 'n_pts' in kwargs.keys():
        n_pts = kwargs['n_pts']
    else:
        # use n in calculation both times
        n_pts = int(kwargs['x_max'] / h)

    # list of x-values
    xs = list(np.linspace(x_0, x_0 + n_pts*h, n_pts))

    # list of corresponding y-values
    ys = [y_0]

    # iterate through
    if kind == 'linear':
        for i in range(n_pts-1):
            # do n_iter steps of Newton root-finding iteration to find the next slope
            # create a function to find zeroes
            big_f = lambda y_m: y_m - ys[i] - h*(q(xs[i+1]) - y_m*p(xs[i+1]))
            big_f_prime = lambda y_m: 1 + h*p(xs[i+1])
            ys.append(root_find(big_f, big_f_prime, ys[i], n_iter))
    else:
        for i in range(n_pts-1):
            big_f = lambda y_m: y_m - ys[i] - h*f(xs[i+1], y_m)
            big_f_prime = lambda y_m: 1 - h*f_prime(xs[i+1], y_m)
            ys.append(root_find(big_f, big_f_prime, ys[i], n_iter))

    return xs, ys


def root_find(f, f_prime, init_guess, steps):
    x = init_guess
    for _ in range(steps):
        x = x - f(x)/f_prime(x)

    return x
