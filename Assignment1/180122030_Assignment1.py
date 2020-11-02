import numpy as np

print("###################### Question 1 ######################")

def f(x):
    return np.exp(-1*x) - np.sin(x)

def df_dx(x):
    return -np.exp(-1*x) - np.cos(x)

left = 0
right = 3

print("Finding the initial guess using Bisection Method")
print(f"Initial range for Bisection Method [{left}, {right}]: f({left}) = {f(left):.5f}; f({right}) = {f(right):.5f};")

assert (f(left) * f(right) < 0)

for i in range(3):
    mid = (left+right)/2
    if (f(mid)*f(left) < 0):
        right = mid
    elif (f(mid)*f(right) < 0):
        left = mid
    else:
        print (f"Solution found at x = {mid:.5f}, f({mid:.5f}) = { f(mid):.5f }")

print(f"After three iterations of Bisection Method, Initial Guess = {mid:.5f}")
print("Begin Newton-Raphson Method")
xn = mid
tol = 1e-5
steps = 0
errors = []

while (True):
    h = - f(xn)/df_dx(xn)
    errors.append(h)
    xn_1 = xn + h
    steps += 1
    print(f"After step {steps}: approximate root = {xn_1:.5f}")

    if (abs(h) < tol):
        print("Tolerance Limit Reached")
        break

    xn = xn_1

p = np.log(np.abs(errors[steps-1]/errors[steps-2])) / np.log(np.abs(errors[steps-2]/errors[steps-3]))
print(f"Approximate order of convergence p = {p:.5f}")
print(f"Number of steps required = {steps}")

print("###################### Question 2 ######################")
print("Solving a non-linear system of equations")
def f1(x,y):
    return np.sin(x*y) + x - y

def f2(x,y):
    return y*np.cos(x*y) + 1

def df1_dx(x,y):
    return y*np.cos(x*y) + 1

def df2_dx(x,y):
    return -1*(y**2) * np.sin(x*y)

def df1_dy(x,y):
    return x*np.cos(x*y) - 1

def df2_dy(x,y):
    return np.cos(x*y) - y*x*np.sin(x*y)

def Jinverse(X):
    """
    Input:
    X : Matrix of size (2,1)
    Returns:
    The inverse of the Jacobian Matrix size (2,2)
    """
    x = X[0,0]
    y = X[1,0]
    det_of_j = df2_dy(x,y)*df1_dx(x,y) - df1_dy(x,y)*df2_dx(x,y)
    jinverse = [
                [df1_dy(x,y),   -1*df1_dy(x,y)],
                [-1*df2_dx(x,y), df1_dx(x,y)]
                ]
    jinverse = np.array(jinverse)
    jinverse /= det_of_j
    return jinverse

def F(X):
    """
    Input:
    X : Matrix of size (2,1)
    Returns:
    A matrix of size (2,1)
    """
    x = X[0,0]
    y = X[1,0]
    _F = [
    [f1(x,y)],
    [f2(x,y)]
    ]
    return np.array(_F)

X_prev = np.array([
                [1],
                [2]
              ])

tol = 1e-3
print(f"First Guess = {X_prev.reshape(-1,)}")
steps = 0
while (True):
    h = -1 * np.matmul(Jinverse(X_prev), F(X_prev))
    X_new = X_prev + h
    steps += 1
    print(f"After step #{steps}, guess = {X_new.reshape(-1,)}")
    if (np.sum(np.abs(h)) < tol):
        print("Tolerance Limit Reached")
        break
    X_prev = X_new

print(f"Approximate solution to the system of equation is [x,y] = {X_new.reshape(-1,)}")
