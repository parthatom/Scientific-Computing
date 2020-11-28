import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def explicit_indexer(h,k,xs,ts):
    x = np.arange(xs[0],xs[1]+h,h)
    t = np.arange(ts[0],ts[1]+k,k)
    return x,t

def plot3d(x,t,z,title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X, Y = np.meshgrid(t,x)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(np.min(z), np.max(z))
    ax.set_xlabel('Time')
    ax.set_ylabel('X cordinate')
    ax.set_zlabel('U(x,t)')
    ax.set_title(title)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(title+'.png')
    plt.show()

def plot_solution(h,k, plot_list):
    x_i,t_j = explicit_indexer(h,k,(0.0,1.0),(0.0,0.1))
    N_x = len(x_i)
    N_t = len(t_j)
    # Initialize values
    U = np.zeros((N_x, N_t))
    U[:,0] = np.sin(np.pi * x_i) # Since index (:,1) indicates U(x,0) ,0<x<1
    # Perform calculation
    r = k/(h*h)
    for j in range(N_t-1):
        for i in range(1,N_x-1):
            U[i][j+1] = r * U[i-1][j] + (1 - 2* r) * U[i][j] + r * U[i+1][j]

    #Exact solution
    Ex = np.zeros_like(U)
    for j in range(N_t):
        for i in range(N_x):
            Ex[i][j] = np.exp(-(np.pi**2)*t_j[j])*np.sin(np.pi*x_i[i])

    if ("Exact" in plot_list):
        plot3d(x_i,t_j,Ex,f'Exact solution for r = {r:.2f}')
    if ("Solution" in plot_list):
        plot3d(x_i,t_j,U,f'Numeric solution for r = {r:.2f}')
    if ("Error" in plot_list):
        plot3d(x_i,t_j,np.abs(U-Ex),f'Error between exact solution and numerical solution for r = {r:.2f}')
    return U, Ex


U, Ex = plot_solution(h = 0.1, k = 0.005, plot_list = [])
avg_err_list = []
a = np.arange(1, 20) * 0.05
for k in a:
    U, Ex = plot_solution(h = 0.1, k = k * 0.01 , plot_list = [])
    avg_err_list.append( np.sqrt(np.mean( np.square(np.abs(U-Ex)) )) )

plt.clf()
plt.close()

plt.plot(a, avg_err_list )
plt.show()
