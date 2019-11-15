# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:55:21 2018
@author: mathemacode
Lorenz RK4 Estimation, then Adams-Bashforth-Moulton integration

Plots 2D and 3D RK4 and 3D A-B-M

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core._multiarray_umath import ndarray


def main():
    rk_lorenz(0, 20, 1000, 9, 9)   # display plots - 2D and 3D for RK (not adams yet) estimation
    adams(0, 20, 1000)             # display plot - Adams-Bashforth-Moulton using RK4


# derivative functions
class derfnc:
    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def x1p(x1, x2, x3):
        return 10 * (x2 - x1)

    def x2p(x1, x2, x3):
        return x1 * (28 - x3) - x2

    def x3p(x1, x2, x3):
        return (x1 * x2) - (8/3)*x3

# A-B-M computation
def adams(a, b, n):
    h = (b-a)/n

    # initialize y1, y2, y3 matrices (and y_predictors)
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)

    y_pred1 = np.zeros(n)
    y_pred2 = np.zeros(n)
    y_pred3 = np.zeros(n)

    # values for Adams-B-M computation
    for E in (0, 1, 2, 3):
        y1[E] = rk_lorenz(a, b, n, E, 1)
        y2[E] = rk_lorenz(a, b, n, E, 2)
        y3[E] = rk_lorenz(a, b, n, E, 3)

    for i in range(4, n):
        # The equations change slightly too, hard to refactor to make this look clean

        # x1p
        y_pred1[i] = y1[i-1] + (h / 24 * (55 *
                                          derfnc.x1p(y1[i - 1], y2[i - 1], y3[i - 1]) - 59 *
                                          derfnc.x1p(y1[i - 2], y2[i - 2], y3[i - 2]) + 37 *
                                          derfnc.x1p(y1[i - 3], y2[i - 3], y3[i - 3]) - 9 * 0))

        y1[i] = y1[i-1] + (h / 24 * (9 *
                                     derfnc.x1p(y_pred1[i - 1], y_pred2[i - 1], y_pred3[i - 1]) + 19 *
                                     derfnc.x1p(y1[i - 1], y2[i - 1], y3[i - 1]) - 5 *
                                     derfnc.x1p(y1[i - 2], y2[i - 2], y3[i - 2]) +
                                     derfnc.x1p(y1[i - 3], y2[i - 3], y3[i - 3])))

        # x2p
        y_pred2[i] = y2[i-1] + (h / 24 * (55 *
                                          derfnc.x2p(y1[i - 1], y2[i - 1], y3[i - 1]) - 59 *
                                          derfnc.x2p(y1[i - 2], y2[i - 2], y3[i - 2]) + 37 *
                                          derfnc.x2p(y1[i - 3], y2[i - 3], y3[i - 3]) - 9 * (15 * (-8) - 15)))

        y2[i] = y2[i-1] + (h / 24 * (9 *
                                     derfnc.x2p(y_pred1[i - 1], y_pred2[i - 1], y_pred3[i - 1]) + 19 *
                                     derfnc.x2p(y1[i - 1], y2[i - 1], y3[i - 1]) - 5 *
                                     derfnc.x2p(y1[i - 2], y2[i - 2], y3[i - 2]) +
                                     derfnc.x2p(y1[i - 3], y2[i - 3], y3[i - 3])))

        # x3p
        y_pred3[i] = y3[i-1] + (h / 24 * (55 *
                                          derfnc.x3p(y1[i - 1], y2[i - 1], y3[i - 1]) - 59 *
                                          derfnc.x3p(y1[i - 2], y2[i - 2], y3[i - 2]) + 37 *
                                          derfnc.x3p(y1[i - 3], y2[i - 3], y3[i - 3]) - 9 * ((15 * 15) - ((8 / 3) * 36))))

        y3[i] = y3[i-1] + (h / 24 * (9 *
                                     derfnc.x3p(y_pred1[i - 1], y_pred2[i - 1], y_pred3[i - 1]) + 19 *
                                     derfnc.x3p(y1[i - 1], y2[i - 1], y3[i - 1]) - 5 *
                                     derfnc.x3p(y1[i - 2], y2[i - 2], y3[i - 2]) +
                                     derfnc.x3p(y1[i - 3], y2[i - 3], y3[i - 3])))

    # 3D plot, because, Lorenz
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(y1, y2, y3, linewidth=1.0, color="red")
    plt.title("Adams-Bashforth-Moutlon Lorenz Estimation")
    plt.style.use('ggplot')
    plt.show()


def rk_lorenz(a, b, n, u, m):
    # a lower bound
    # b upper bound
    # n number of steps
    # j indicator for which Y1, Y2, Y3 val is needed to be return by function
    # OR, can use u == 9 and m == 9 to plot in 2D or 3D
    # m means x1, x2, x3 when calling for values Y1, Y2, Y3, etc...

    h = (b-a)/n  # step size
    p = 0  # flag for while loop

    # intialize lists for plot
    t = np.zeros(int(n+1))
    x1 = np.zeros(int(n+1))
    x2 = np.zeros(int(n+1))
    x3 = np.zeros(int(n+1))

    # define initial values
    t[0] = 0
    x1[0] = 15
    x2[0] = 15
    x3[0] = 36

    # initialize slots for values of RK4
    i = np.zeros(4)  # for x1
    j = np.zeros(4)  # for x2
    k = np.zeros(4)  # for x3

    # iterate n times
    while p < n:

        ''' RK4 Calcuations, adapted into Numpy arrays '''
        i[0] = h * derfnc.x1p(x1[p], x2[p], x3[p])
        j[0] = h * derfnc.x2p(x1[p], x2[p], x3[p])
        k[0] = h * derfnc.x3p(x1[p], x2[p], x3[p])

        i[1] = h * derfnc.x1p(x1[p] + (1/2)*i[0], x2[p] + (1/2)*j[0], x3[p] + (1/2)*k[0])
        j[1] = h * derfnc.x2p(x1[p] + (1/2)*i[0], x2[p] + (1/2)*j[0], x3[p] + (1/2)*k[0])
        k[1] = h * derfnc.x3p(x1[p] + (1/2)*i[0], x2[p] + (1/2)*j[0], x3[p] + (1/2)*k[0])

        i[2] = h * derfnc.x1p(x1[p] + (1/2)*i[1], x2[p] + (1/2)*j[1], x3[p] + (1/2)*k[1])
        j[2] = h * derfnc.x2p(x1[p] + (1/2)*i[1], x2[p] + (1/2)*j[1], x3[p] + (1/2)*k[1])
        k[2] = h * derfnc.x3p(x1[p] + (1/2)*i[1], x2[p] + (1/2)*j[1], x3[p] + (1/2)*k[1])

        i[3] = h * derfnc.x1p(x1[p] + i[2], x2[p] + j[2], x3[p] + k[2])
        j[3] = h * derfnc.x2p(x1[p] + i[2], x2[p] + j[2], x3[p] + k[2])
        k[3] = h * derfnc.x3p(x1[p] + i[2], x2[p] + j[2], x3[p] + k[2])

        x1[p + 1] = x1[p] + (1/6) * (i[0] + (2*i[1]) + (2*i[2]) + i[3])
        x2[p + 1] = x2[p] + (1/6) * (j[0] + (2*j[1]) + (2*j[2]) + j[3])
        x3[p + 1] = x3[p] + (1/6) * (k[0] + (2*k[1]) + (2*k[2]) + k[3])

        t[p + 1] = t[p] + h

        p += 1  # advance flag variable

    ''' Y1, Y2, Y3, and options for Adams-Bashforth-Moulton Method '''
    if u == 9 and m == 9:

        ''' Plot Results of RK4 '''
        # 2D plot to show consistency of estimations
        plt.plot(t, x1, '-', label="X1 RK4", linewidth=2.0)
        plt.plot(t, x2, '-', label="X2 RK4", linewidth=2.0)
        plt.plot(t, x3, '-', label="X3 RK4", linewidth=2.0)

        plt.xlim(a, b)
        plt.title("RK4 Method Result: Lorenz Problem")
        plt.xlabel("t")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        # 3D plot, because, Lorenz
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x1, x2, x3, linewidth=1.0, color="blue")
        plt.title("RK4 Lorenz Estimation")
        plt.style.use('ggplot')
        plt.show()

    elif u in (0, 1, 2, 3):
        if m == 1:
            return x1[u]
        elif m == 2:
            return x2[u]
        elif m == 3:
            return x3[u]
        else:
            return


if __name__ == '__main__':
    main()
else:
    print("Script imported - run rk_lorenz() or adams()")
