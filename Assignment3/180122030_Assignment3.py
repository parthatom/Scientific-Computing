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

print("N\tR_rule\tT_rule\tS_rule\tE_R\tE_T\tE_S")
for i in range(len(N)):
    n = N[i]
    interval_list = get_interval(n)
    i_r = rectangle_rule(interval_list)
    i_t = trapezium_rule(interval_list)
    i_s = simpson_rule(interval_list)
    e_r = np.abs(i_r - i_actual)
    e_s = np.abs(i_s - i_actual)
    e_t = np.abs(i_t - i_actual)
    print(f"{n}\t{i_r:.5f}\t{i_t:.5f}\t{i_s:.5f}\t{e_r:.5f}\t{e_t:.5f}\t{e_s:.5f}")
