import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

print("============ Question 1 ============")

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
    plt.savefig("180122030_A4_Q1_" + title + '.jpg')
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
        plot3d(x_i,t_j,np.abs(U-Ex),f'Error for r = {r:.2f}')
    return U, Ex

U, Ex = plot_solution(h = 0.1, k = 0.005, plot_list = ['Exact', 'Solution', 'Error'])

avg_err_list = []
r_list = np.arange(1, 21) * 0.05
for i in r_list:
    U, Ex = plot_solution(h = 0.1, k = i * 0.01 , plot_list = [])
    avg_err_list.append( np.mean(np.abs(U-Ex)) )

plt.plot(r_list, avg_err_list )
plt.title("Mean Absolute Error for various values of r")
plt.xlabel("r")
plt.ylabel("Mean of Absolute Error")
plt.savefig("180122030_A4_Q1_MAE_vs_R.jpg")
plt.show()
plt.clf()
plt.close()

print("We can see that after r = 0.5, the error is increasing rapidly. \n It's interesting also to observe that uptil r = 0.15, it decreases")

for k in [0.005, 0.007, 0.01]:
    U, Ex = plot_solution(h = 0.1, k = k , plot_list = ['Error'])

print("============ Question 2 ============")

def initial_value_condition_1(x):
  """
  Initial value condition for x in [0, 1/2]
  """
  return 2*x

def initial_value_condition_2(x):
  """
  Initial value condition for x in [1/2, 1]
  """
  return 2*(1-x)

t = 0.2
h = 0.1
k = 0.01
n = round((t/k)+1)

def crank_nicolson_method(r):
  """
  Crank Nicolson Method
  """
  array = np.zeros([11, n])
  array[:6, 0] = initial_value_condition_1(np.linspace(0, 0.5, 6, endpoint=True))
  array[6:, 0] = initial_value_condition_2(np.linspace(0.6, 1, 5, endpoint=True))
  array[0, :] = 0
  array[10, :] = 0
  B = np.zeros([9, 9])
  C = np.zeros([9, 9])
  for i in range(0, 9):
    if i == 0:
      B[i, i] = 2+2*r
      B[i, i+1] = -1*r
      C[i, i] = 0
      C[i, i+1] = r
    elif i == 8:
      B[i, i] = 2+2*r
      B[i, i-1] = -1*r
      C[i, i] = 0
      C[i, i-1] = r
    else:
      B[i, i] = 2+2*r
      B[i, i+1] = -1*r
      B[i, i-1] = -1*r
      C[i, i] = 0
      C[i, i+1] = r
      C[i, i-1] = r

  for j in range(1, n):
    array[1:10, j] = (np.linalg.inv(B) @ C @ array[1:10, j-1].T).T# '@' denotes matrix multiplication
  return array

r = k/h**2
A = crank_nicolson_method(r)
x1 = np.linspace(0, 1, 11, endpoint=True)
t1 = np.linspace(0, 0.2, n, endpoint=True)
T1, X1 = np.meshgrid(t1, x1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(T1, X1, A, rstride=1, cstride=1, cmap='magma', edgecolor='none')
ax.set_title('Predicted solution');
ax.set_zlabel("Temperature")
ax.set_ylabel("X")
ax.set_xlabel("Time")
ax.view_init(20, 45)
plt.show()
plt.savefig("180122030_Assignment4_Q2.jpg")
plt.clf()
plt.close()

print("============ Question 3 ============")

def boundary_function(x, y):
  """
  Boundary Function
  """
  return np.sin(x)*np.sin(y)

t = 1
k = 0.01
h = 0.2
n = int (np.round((t/k)+1))

def explicit_scheme(r):
  array = np.zeros([6, 6, n])
  X, Y = np.meshgrid(np.linspace(0, 1, 6, endpoint=True), np.linspace(0, 1, 6, endpoint=True))
  array[:, :, 0] = boundary_function(X, Y)
  for i in range(1, 100):
    for j in range(1, 5):
      for k in range(1, 5):
        array[j, k, i] = r * array[j+1, k, i-1] + (1 - 4*r) * array[j, k, i-1] + r * array[j-1, k, i-1] + r * array[j, k+1, i-1] + r * array[j, k-1, i-1]
  return array

r = 0.25
array = explicit_scheme(r)

X, Y = np.meshgrid(np.linspace(0, 1, 6, endpoint=True), np.linspace(0, 1, 6, endpoint=True))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 10), subplot_kw={'projection': '3d'})

ax_list = [ax1, ax2, ax3, ax4]
t_list = [0, 1, 2, 9] #List of Temperatures to Display

for i in range(4):
    ax_list[i].plot_surface(X, Y, array[:, :, t_list[i]], rstride=1, cstride=1,
                    cmap='magma', edgecolor='none')
    ax_list[i].set_title(f'At t= {t_list[i] * 0.01}');
    ax_list[i].view_init(24, -65) #Orientation
    ax_list[i].set_zlim(0, 0.7)
    ax_list[i].set_zlabel("Temperature")
    ax_list[i].set_xlabel("X")
    ax_list[i].set_ylabel("Y")

fig.suptitle("Solution Prediction at Different time stamps")
plt.show()
plt.savefig("180122030_Assignment4_Q3.jpg")
plt.clf()
