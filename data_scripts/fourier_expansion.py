import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

"""Fourier series expansion of of $2L$-periodic function $f:\mathbb{R}\rightarrow \mathbb{R}$
f(x) = \dfrac{1}{2} a_0 + \sum_{n=1}^\infty a_n \cos(\dfrac{n \pi x}{L}) + \sum_{n=1}^\infty b_n \sin(\dfrac{n \pi x}{L})$ <br>
Fourier coefficients
a_0 = \dfrac{1}{2} \int_0^{2L}f(x)dx$
a_n = \dfrac{1}{L} \int_0^{2L}f(x)\cos(\dfrac{n \pi x}{L})dx$
b_n = \dfrac{1}{L} \int_0^{2L}f(x)\sin(\dfrac{n \pi x}{L})dx$
for any $n$ positive integer.
See https://mathworld.wolfram.com/FourierSeries.html eqs 17-18-19"""


def evaluate_fourier_series(x, L, fourier_coefs):
    """
    Evaluate Fourier series on domain [0,2L]
    :param x: interval of points
    :type x: np.array
    :param L: half of function period
    :type L: float
    :param fourier_coefs: coefficients a0; a1, ..., an (cos); b1, ..., bn (sin)
    :type fourier_coefs: dict
    :return: function evaluated on given domain
    :rtype: np.array
    """
    ys = []
    for point in x:
        y = 0
        for coef in fourier_coefs.keys():
            if '0' in coef:
                y += fourier_coefs[coef]/2
            elif 'a' in coef:
                y += fourier_coefs[coef] * np.cos(point * np.pi * float(coef[1:])/L)
            elif 'b' in coef:
                y += fourier_coefs[coef] * np.sin(point * np.pi * float(coef[1:])/L)
        ys.append(y)
    return ys


def get_fourier_coefficients(x, y, fourier_degree, diagnostic=False):
    """
    Approximate periodic function as Fourier series of given degree.
    The function is specified by evaluation on an equispaced one-dimensional interval with length equal to period of the
     given function.
    The Fourier coefficients are expressed in cos/sin basis, see https://mathworld.wolfram.com/FourierSeries.html
    eqs 17-18-19.
    The domain interval must be of equispaced points
    :param x: interval of equispaced points
    :type x: np.array
    :param y: function values on domain interval
    :type y: np.array
    :param fourier_degree: ...
    :type fourier_degree: int
    :param diagnostic: whether to plot original vs approximated function
    :type diagnostic: bool
    :return: dictionary of Fourier coefficients
    :rtype: dict
    """

    if np.abs(y[0] - y[-1]) > 10 ** -5:
        raise ValueError('Given function is not periodic!')

    # Get interval and subinterval lengths
    int_len = np.abs(x[-1] - x[0])
    L = np.abs(int_len / 2)
    subint_len = x[1] - x[0]
    coefs = {}

    # To compute Fourier coefficients we need to compute integrals. Romberg method for integration is more accurate
    # than trapezoid or Simpson when function is evaluated on 2^k + 1 equispaced points
    if np.log2(len(x) - 1) % 1 == 0:
        integr_alg = 'romberg'
    else:
        integr_alg = 'simpson'

    # Compute Fourier coefficients
    if integr_alg == 'romberg':
        c = integrate.romb(y, dx=subint_len) / L
    else:
        c = integrate.simpson(y, x) / L
    coefs['a0'] = c

    for i in range(1, fourier_degree + 1):
        if integr_alg == 'romberg':
            a = integrate.romb(y * np.cos((np.pi * i * x) / L), dx=subint_len) / L
            b = integrate.romb(y * np.sin((np.pi * i * x) / L), dx=subint_len) / L
        else:
            a = integrate.simpson(y * np.cos((np.pi * i * x) / L), x) / L
            b = integrate.simpson(y * np.sin((np.pi * i * x) / L), x) / L
        coefs[f'a{i}'] = a
        coefs[f'b{i}'] = b

    if diagnostic:
        # Evaluate Fourier approximation on given interval
        y_approx = evaluate_fourier_series(x=x, L=L, fourier_coefs=coefs)
        # Plot original and approximated functions
        fig = plt.figure()
        plt.plot(x, y, label='original function')
        plt.plot(x, y_approx, label='approximated function')
        plt.title('Fourier approximation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        fig.show()

    return coefs
