import numpy as np
import matplotlib.pyplot as plt

print("################ Question 1: ################")
def f1(x):
    """
    The function to approximate.
    Runge's Example Function
    """
    return 1/(1+x*x)

def xi(i):
    """
    Gives the value of the ith - x coordinate
    """
    return -5 + 10*i/8

xis = xi(np.arange(1,9)) # Array of x-coordinates
fis = f1(xis) #Array of f(xi)

l_den = [] # Denominators of Li's
for i in range(8):
    den = 1
    for j in range(8):
        if (i==j):
            continue
        den *= (xis[i] - xis[j])
    l_den.append(den)
l_den = np.array(l_den)

def p1(x):
    """
    Interpolating function
    Returns: The value of the Interpolating function at x
    """

    l_num = []
    for i in range(8):
        num = 1
        for j in range(8):
            if (i==j):
                continue
            num *= (x - xis[j])
        l_num.append(num)
    l_num = np.array(l_num)
    lis = l_num/l_den
    p_of_x = np.dot(fis, lis) # Value at x
    return p_of_x

# for i in range(8):
    # print(f1(xis[i]), p1(xis[i]), f1(xis[i])== p1(xis[i]))

plot_xs = np.linspace(-4, 5, 100)
plot_fxs = f1(plot_xs)

plot_pxs = []
for x in plot_xs:
    plot_pxs.append(p1(x))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(plot_xs, plot_pxs, 'b', label = 'p(x)')
plt.plot(plot_xs, plot_fxs, 'r', label = 'f(x)')
plt.legend(loc='lower right')
plt.savefig("plot_q1.png")

print("################ Question 2: ################")
def f2(x):
    return x + 2/x

n = 3
xis_2 = ((1.0/2.0) * np.arange(1,5)).reshape(-1,1)
fis_2 = f2(xis_2).reshape(-1,1)

his = xis_2[1:n+1] - xis_2[0:n]
uis = 2 * (his[1:n] + his[0:n-1])
bis = 6 * (fis_2[1:n+1] - fis_2[0:n])/his
vis = bis[1:n] - bis[0:n-1]

#Am = v ==> m = A-1v
A = [
    [uis[0], his[1]],
    [his[1], uis[1]]
    ]
A = np.array(A).reshape(2,2)
A_inv = np.linalg.inv(A)
# A_inv = (np.array(A_inv)/(uis[1]*uis[0] - his[1]*his[1])).reshape(2,2)
mis = np.dot(A_inv, vis)

zs = np.zeros((4,1))
zs[1:3] = mis
mis = zs
print(A_inv, mis,  vis)
def Si(x, i):
    val = (mis[i]/(6*his[i])) * ((xis_2[i+1]-x)**3) + \
          (mis[i+1]/(6*his[i])) * ((x-xis_2[i])**3) + \
          (fis_2[i+1]/his[i] - (mis[i+1]*his[i])/6.0) * (x - xis_2[i]) + \
          (fis_2[i]/his[i] - (mis[i]*his[i])/6.0) * (xis_2[i+1] - x)
    return val

plot_xs_2 = np.linspace(0.5, 2, 50)
plot_fxs_2 = f2(plot_xs_2)

plot_pxs_2 = []
for x in plot_xs_2:
    for i in range(3):
        if (xis_2[i] <= x and xis_2[i+1] >= x ):
            plot_pxs_2.append(Si(x,i))
            break

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(plot_xs_2, plot_pxs_2, 'b', label = 'S(x)')
plt.plot(plot_xs_2, plot_fxs_2, 'r', label = 'f(x)')
plt.legend(loc='lower right')
plt.savefig("plot_q2.png")
plt.show()
