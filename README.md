# Adams-Bashforth-Moulton_LORENZ
Using the Adams-Bashforth-Moulton method (via Runge-Kutta 4th order) to approximate the Lorenz problem.  Firstly starting with RK4 alone to see how the accuracy compares before implementing ABM.  ABM then uses RK4 as part of its computation.  I ran ABM up to n=1,000,000.

## RK 4 Solo Solution: n=100,000
![RK4_2D](https://raw.githubusercontent.com/mathemacode/Adams-Bashforth-Moulton_LORENZ/master/plots/2D_plot.png)
![RK4](https://github.com/mathemacode/Adams-Bashforth-Moulton_LORENZ/blob/master/plots/RK4_est.png?raw=true)

## Adams-Bashforth-Moulton Solution, n=1,000,000
![ABM](https://github.com/mathemacode/Adams-Bashforth-Moulton_LORENZ/blob/master/plots/AdamsBM_est.png?raw=true)
