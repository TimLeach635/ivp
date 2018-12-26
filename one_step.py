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


def generate_runge_kutta(a, b):
    """Generate the Runge-Kutta method from the given Butcher table.

    The Butcher table takes this form:

     c | a
    ___|___
       |  T
       | b

    with 'a' an n x n square matrix, and 'b' an n x 1 column vector, and 'c', another column vector,
    the row sums of 'a'.

    Parameters:
        a - a scalar, or n x n array.
        b - a scalar, or n x 1 array.

    Returns:
        A function with the following signature:
        xs, ys = rk_method(x_0, y_0, h, **kwargs)

        Positional arguments:
            x_0 and y_0 are the given initial values: y(x_0) = y_0.
            h is the grid spacing.

        Keyword arguments:
            either define 'n_pts' or 'x_max'.
            n_pts is the number of grid points (not including the origin), and
            x_max is the x-value of the largest grid point.

            either define 'f', or 'p' and 'q'.
            if 'f' is defined, it should be a 2-argument function, and the method solves the nonlinear ODE
            dy/dx = f(x, y).
            if the R-K method you define is implicit, you must also define 'f_prime', the derivative with respect to y
            of 'f'.
            if 'p' and 'q' are defined, they should both be 1-argument functions, and the method will solve the linear
            ODE dy/dx + p(x)*y = q(x).

            define 'n_iter' if the R-K method defined is implicit.
            it represents the number of iteration steps in the method used to solve the implicit bit.

        Returns:
            xs and ys, two numpy arrays holding the x and y coordinates of the numerical solution.
            matplotlib-able.
    """
    # is it just scalars?
    try:
        a = float(a)
        b = float(b)
        # it's got this far so they're all scalar
        n_stages = 1
        c = a
    except TypeError:
        # they are arrays, check size compatibility
        # check dimensionality
        if not (a.ndim == b.ndim == 2):
            raise TypeError("'a' and 'b' should both be 2-dimensional.")

        # then, ensure they are both column vectors
        if not b.shape[1] == 1:
            raise TypeError("'b' should be a column vectors.")

        # number of stages is then this
        n_stages = b.shape[0]

        # make sure a fits also
        if not (a.shape[0] == a.shape[1] == n_stages):
            raise TypeError("'a' must be square and compatible with 'b'.")

        # set c
        c = a.sum(1)

    # explicit or implicit?
    explicit = False
    if n_stages == 1:
        if a == 0:
            explicit = True
    else:
        # check strictly lower triangular
        if np.allclose(a, np.tril(a, -1)):
            explicit = True

    # form function
    if explicit:
        if n_stages == 1:
            def rk_method(x_0, y_0, h, **kwargs):
                """Solve the given one-dimensional ODE.

                :param x_0:
                :param y_0:
                :param h:
                :param kwargs:
                :return (xs, ys): two numpy arrays of the same length representing the numerical solution.
                """

                # linear or nonlinear
                if 'p' in kwargs.keys() and 'q' in kwargs.keys():
                    linear = True
                    p = kwargs['p']
                    q = kwargs['q']
                elif 'f' in kwargs.keys():
                    linear = False
                    f = kwargs['f']
                else:
                    raise TypeError("You must pass keyword arguments 'p' and 'q', or 'f'.")

                # number of points
                if 'n_pts' in kwargs.keys():
                    n_pts = kwargs['n_pts']
                elif 'x_max' in kwargs.keys():
                    n_pts = int(kwargs['x_max']/h)
                else:
                    raise TypeError("You must pass either keyword arguments 'n_pts' or 'x_max'.")

                # form output
                xs = np.linspace(x_0, x_0 + h*n_pts, n_pts)
                ys = [y_0]

                # solve
                if linear:
                    for i in range(n_pts-1):
                        ys.append(ys[i] + h*b*(q(xs[i]) - p(xs[i])*ys[i]))
                else:
                    for i in range(n_pts-1):
                        ys.append(ys[i] + h*b*f(xs[i], ys[i]))

                return xs, np.array(ys)
        else:
            def rk_method(x_0, y_0, h, **kwargs):
                """

                :param x_0:
                :param y_0:
                :param h:
                :param kwargs:
                :return (xs, ys):
                """

                # linear or nonlinear
                if 'p' in kwargs.keys() and 'q' in kwargs.keys():
                    linear = True
                    p = kwargs['p']
                    q = kwargs['q']
                elif 'f' in kwargs.keys():
                    linear = False
                    f = kwargs['f']
                else:
                    raise TypeError("You must pass keyword arguments 'p' and 'q', or 'f'.")

                # number of points
                if 'n_pts' in kwargs.keys():
                    n_pts = kwargs['n_pts']
                elif 'x_max' in kwargs.keys():
                    n_pts = int(kwargs['x_max']/h)
                else:
                    raise TypeError("You must pass either keyword arguments 'n_pts' or 'x_max'.")

                # form output
                xs = np.linspace(x_0, x_0 + h * n_pts, n_pts)
                ys = [y_0]

                # solve
                if linear:
                    for m in range(n_pts-1):
                        y_next = ys[m]
                        ks = []
                        for i in range(n_stages):
                            # find k_i
                            # to do that, find the y co-ordinate
                            y_k = ys[m]
                            for j in range(i):  # we know it's explicit, so we don't need to iterate over the zeros
                                y_k += h*a[i, j]*ks[j]
                            ks.append(q(xs[m] + h*c[i]) - y_k*p(xs[m] + h*c[i]))
                            y_next += h*b[i, 0]*ks[i]
                        ys.append(y_next)

                return xs, np.array(ys)

    # return function
    return rk_method


def root_find(f, f_prime, init_guess, steps):
    x = init_guess
    for _ in range(steps):
        x = x - f(x)/f_prime(x)

    return x
