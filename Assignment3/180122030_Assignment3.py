import numpy as np
import pandas as pd

# Make sure you have them both installed
# if you don't, run `pip install numpy` and `pip install pandas`

print("######## QUESTION: 1 ########")

def f(x):
    return np.exp(-np.power(x, 2))

def get_interval(n):
    """
    Returns n intervals between the range [0,1]
    """
    l = np.arange(n+1)/n
    interval_list = []
    for i in range(len(l)-1):
        interval_list.append([l[i], l[i+1]])
    return interval_list

def rectangle_rule(interval_list):
    integral = 0
    for interval in interval_list:
        xi = interval[1]   #Xi
        xi_1 = interval[0] #Xi-1
        h = xi - xi_1
        integral += f(xi_1)*h
    return integral

def trapezium_rule(interval_list):
    integral = 0
    sum = 0
    for i, interval in enumerate(interval_list):
        xi = interval[1]   #Xi
        xi_1 = interval[0] #Xi-1
        h = xi - xi_1
        integral += (h/2.0) * (f(xi_1) + f(xi))
    return integral

def simpson_rule(interval_list):
    integral = 0
    for interval in interval_list:
        xi = interval[1]   #Xi
        xi_1 = interval[0] #Xi-1
        h = xi - xi_1
        integral += (h/6.0) * ( f(xi_1) + 4*f(xi-h/2.0) + f(xi) )
    return integral

i_actual = 0.7468241
N = [50, 100, 200]

print("N\tR_rule\t\tT_rule\t\tS_rule\t\tE_R\t\tE_T\t\tE_S")
for i in range(len(N)):
    n = N[i]
    interval_list = get_interval(n)
    i_r = rectangle_rule(interval_list)
    i_t = trapezium_rule(interval_list)
    i_s = simpson_rule(interval_list)
    e_r = np.abs(i_r - i_actual)
    e_s = np.abs(i_s - i_actual)
    e_t = np.abs(i_t - i_actual)
    print(f"{n:.1f}\t{i_r:.5f}\t\t{i_t:.5f}\t\t{i_s:.5f}\t\t{e_r:.5f}\t\t{e_t:.5f}\t\t{e_s:.5f}")

print("######## QUESTION: 2 ########")

def f2(x, y1, y2):
    """
    x0: Initial point
    y1: y(x0)
    y2: y'(x0)
    """
    return y2

def g2(x,y1, y2):
    """
    x0: Initial point
    y1: y(x0)
    y2: y'(x0)
    """
    return np.cos(x*y1)


def F(x, Y):
    """
    x: Initial point
    Y: 2-dim Vector
    Returns: The value of vector function at x and Y
    """
    return np.array([f2(x, Y[0], Y[1]), g2(x, Y[0], Y[1])])

N = 64

def runge_kutta(x0, y1, y2, xn):
    """
    x0: Initial point
    y1: y(x0)
    y2: y'(x0)
    """
    x_list = [x0]
    Y = np.array([y1, y2])
    Y_list = [Y]
    h = (xn - x0)/N
    # print(x0, Y)
    for n in range(N):
        xn = x_list[n]
        Yn = Y_list[n]

        K1 = h * F(xn, Yn)
        K2 = h * F(xn +  h/2, Yn + (1/2)*K1)
        K3 = h * F(xn + h/2, Yn + (1/2)*K2)
        K4 = h * F(xn + h, Yn + K3)

        x_new = xn + h
        Y_new = Yn + (1/6) * (K1 + 2*K2 + 2*K3 + K4)
        # print(x_new, Y_new)
        x_list.append(x_new)
        Y_list.append(Y_new)
    Yn = Y_list[N-1]

    return Yn[0],Yn[1]

x_ns = [0.25, 0.5, 0.75, 1]
x_0 = 0
y1 = 1
y2 = 0
print(f"x_n\t||\ty( x_n )\t||\ty'(x_n)")

for x_n in x_ns:
    y_n, y_prime_n = runge_kutta(x_0, y1, y2, x_n)
    print(f"{x_n}\t||\t{y_n:.6f}\t||\t{y_prime_n:.6f}")
