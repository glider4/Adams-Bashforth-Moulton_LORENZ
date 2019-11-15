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

def main():
    adams(0, 20, 1000)


def adams(a, b, n):
    RK_lorenz(0, 20, 20000, 9, 9)  # display plots - 2D and 3D for RK (not adams yet) estimation
   
    h = (b-a)/n
    p = 0
    
    # derivative functions, NO t
    def x1p(x1, x2, x3):
        return 10 * (x2 - x1)


    def x2p(x1, x2, x3):
        return x1 * (28 - x3) - x2


    def x3p(x1, x2, x3):
        return (x1 * x2) - (8/3)*x3
        
    # initialize Y1, Y2, Y3 matrices (and predictors)
    Y1 = np.zeros(n)
    Y2 = np.zeros(n)
    Y3 = np.zeros(n)
    
    Y_pred1 = np.zeros(n)
    Y_pred2 = np.zeros(n)
    Y_pred3 = np.zeros(n)
    
    # values for Adams-B-M computation
    for E in (0,1,2,3):
        Y1[E] = RK_lorenz(a, b, n, E, 1)
        Y2[E] = RK_lorenz(a, b, n, E, 2)
        Y3[E] = RK_lorenz(a, b, n, E, 3)
    
    # define functions to be called in Adams-B-M iterations below
    def F(R, x1, x2, x3):
        if R == 1:
            return x1p(x1, x2, x3)
        elif R == 2:
            return x2p(x1, x2, x3)
        elif R == 3:
            return x3p(x1, x2, x3)
        else:
            return 0


    for i in range(4,n):
        p = 1
        Y_pred1[i] = Y1[i-1] + ( h/24 * ( 55*F(p,Y1[i-1],Y2[i-1],Y3[i-1]) - 59*F(p,Y1[i-2],Y2[i-2], Y3[i-2]) + 37*F(p,Y1[i-3],Y2[i-3],Y3[i-3]) - 9*0 ))
        Y1[i] = Y1[i-1] + ( h/24 * ( 9*F(p,Y_pred1[i-1],Y_pred2[i-1],Y_pred3[i-1]) + 19*F(p,Y1[i-1],Y2[i-1], Y3[i-1]) - 5*F(p,Y1[i-2],Y2[i-2], Y3[i-2]) + F(p,Y1[i-3],Y2[i-3],Y3[i-3]) ))
        
        p = 2
        Y_pred2[i] = Y2[i-1] + ( h/24 * ( 55*F(p,Y1[i-1],Y2[i-1],Y3[i-1]) - 59*F(p,Y1[i-2],Y2[i-2], Y3[i-2]) + 37*F(p,Y1[i-3],Y2[i-3],Y3[i-3]) - 9*(15*(-8) - 15)))
        Y2[i] = Y2[i-1] + ( h/24 * ( 9*F(p,Y_pred1[i-1],Y_pred2[i-1],Y_pred3[i-1]) + 19*F(p,Y1[i-1],Y2[i-1], Y3[i-1]) - 5*F(p,Y1[i-2],Y2[i-2], Y3[i-2]) + F(p,Y1[i-3],Y2[i-3],Y3[i-3]) ))
        
        p = 3
        Y_pred3[i] = Y3[i-1] + ( h/24 * ( 55*F(p,Y1[i-1],Y2[i-1],Y3[i-1]) - 59*F(p,Y1[i-2],Y2[i-2], Y3[i-2]) + 37*F(p,Y1[i-3],Y2[i-3],Y3[i-3]) - 9*((15*15) - ((8/3)*36) ) ))
        Y3[i] = Y3[i-1] + ( h/24 * ( 9*F(p,Y_pred1[i-1],Y_pred2[i-1],Y_pred3[i-1]) + 19*F(p,Y1[i-1],Y2[i-1], Y3[i-1]) - 5*F(p,Y1[i-2],Y2[i-2], Y3[i-2]) + F(p,Y1[i-3],Y2[i-3],Y3[i-3]) ))
    
    # 3D plot, because, Lorenz
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(Y1, Y2, Y3, linewidth = 1.0, color = "red")
    plt.title("Adams-Bashforth-Moutlon Lorenz Estimation")
    plt.style.use('ggplot')
    plt.show()


def RK_lorenz(a, b, n, Y, E):
    # a lower bound
    # b upper bound
    # n number of steps
    # Y indicator for which Y1, Y2, Y3 val is needed to be return by function
    # OR, can use Y == 9 and E == 9 to plot in 2D or 3D
    # E means x1, x2, x3 when calling for values Y1, Y2, Y3, etc...
    
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
    
    # derivatives (p for prime)
    def x1p(t, x1, x2, x3):
        return 10 * (x2 - x1)


    def x2p(t, x1, x2, x3):
        return x1 * (28 - x3) - x2


    def x3p(t, x1, x2, x3):
        return (x1 * x2) - (8/3)*x3
    
    # iterate n times
    while p < n:
        
        ''' RK4 Calcuations, adapted into Numpy arrays '''
        i[0] = (h) * x1p(t[p], x1[p], x2[p], x3[p])
        j[0] = (h) * x2p(t[p], x1[p], x2[p], x3[p])
        k[0] = (h) * x3p(t[p], x1[p], x2[p], x3[p])
        
        i[1] = (h) * x1p(t[p] + (h/2), x1[p] + (1/2)*i[0], x2[p] + (1/2)*j[0], x3[p] + (1/2)*k[0])
        j[1] = (h) * x2p(t[p] + (h/2), x1[p] + (1/2)*i[0], x2[p] + (1/2)*j[0], x3[p] + (1/2)*k[0])
        k[1] = (h) * x3p(t[p] + (h/2), x1[p] + (1/2)*i[0], x2[p] + (1/2)*j[0], x3[p] + (1/2)*k[0])
        
        i[2] = (h) * x1p(t[p] + (h/2), x1[p] + (1/2)*i[1], x2[p] + (1/2)*j[1], x3[p] + (1/2)*k[1])
        j[2] = (h) * x2p(t[p] + (h/2), x1[p] + (1/2)*i[1], x2[p] + (1/2)*j[1], x3[p] + (1/2)*k[1])
        k[2] = (h) * x3p(t[p] + (h/2), x1[p] + (1/2)*i[1], x2[p] + (1/2)*j[1], x3[p] + (1/2)*k[1])
        
        i[3] = (h) * x1p(t[p] + (h), x1[p] + i[2], x2[p] + j[2], x3[p] + k[2])
        j[3] = (h) * x2p(t[p] + (h), x1[p] + i[2], x2[p] + j[2], x3[p] + k[2])
        k[3] = (h) * x3p(t[p] + (h), x1[p] + i[2], x2[p] + j[2], x3[p] + k[2])
        
        x1[p + 1] = x1[p] + (1/6) * ( i[0] + (2*i[1]) + (2*i[2]) + i[3] )
        x2[p + 1] = x2[p] + (1/6) * ( j[0] + (2*j[1]) + (2*j[2]) + j[3] )
        x3[p + 1] = x3[p] + (1/6) * ( k[0] + (2*k[1]) + (2*k[2]) + k[3] )
        
        t[p + 1] = t[p] + h
        
        p += 1  # advance flag variable
    
      
    ''' Y1, Y2, Y3, and options for Adams-Bashforth-Moulton Method '''
    if Y == 9 & E == 9:
        
        ''' Plot Results of RK4 '''
        # 2D plot to show consistency of estimations
        plt.plot(t, x1, '-', label = "X1 RK4", linewidth = 2.0)
        plt.plot(t, x2, '-', label = "X2 RK4", linewidth = 2.0)
        plt.plot(t, x3, '-', label = "X3 RK4", linewidth = 2.0)
        
        plt.xlim(a, b)
        plt.title("RK4 Method Result: Lorenz Problem")
        plt.xlabel("t")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
        
        # 3D plot, because, Lorenz
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x1, x2, x3, linewidth = 1.0, color = "blue")
        plt.title("RK4 Lorenz Estimation")
        plt.style.use('ggplot')
        plt.show()
    
    elif Y in (0,1,2,3):
        if E == 1:
            return x1[Y]
        elif E == 2:
            return x2[Y]
        elif E == 3:
            return x3[Y]
        else:
            return


if __name__ == '__main__':
    main()
