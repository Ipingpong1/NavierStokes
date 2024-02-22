import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib.colors import Normalize

s = 3
plot_every = 2
obstacle = 1

Ly = 3
Lx = 3

Ny = 41*s
if Ly > Lx:
    Ny *= np.abs(Ly-Lx)
Nx = 41*s
if Lx > Ly:
    Nx *= np.abs(Lx-Ly)

Nt = 9000
Nit = 50

# -0.09282363336737229

rho = 1
nu = .1
inletVelocity = -3
lidVelocity = 0

x_vec = np.linspace(0, Ly, Ny)
y_vec = np.linspace(0, Lx, Nx)
X, Y = np.meshgrid(x_vec, y_vec)

dx = x_vec[2] - x_vec[1]
dy = y_vec[2] - y_vec[1]
dt = .001

u = np.zeros([len(x_vec), len(y_vec)])
v = np.zeros([len(x_vec), len(y_vec)])
p = np.zeros([len(x_vec), len(y_vec)])

def fd_x(a):
    return ( ( a[1:-1, 1:-1] - a[:-2, 1:-1] ) / dx)
def fd_y(a):
    return ( ( a[1:-1, 1:-1] - a[1:-1, :-2] ) / dy)
def fd_xx(a):
    return ( ( a[2:, 1:-1] - 2*a[1:-1, 1:-1] + a[:-2, 1:-1] ) / (dx**2) )
def fd_yy(a):
    return ( ( a[1:-1, 2:] - 2*a[1:-1, 1:-1] + a[1:-1, :-2] ) / (dy**2) )
def cd_x(a):
    return ( ( a[2:, 1:-1] - a[:-2, 1:-1] ) / (2*dx) )
def cd_y(a):
    return ( ( a[1:-1, 2:] - a[1:-1, :-2] ) / (2*dy) )

b = np.zeros([len(x_vec), len(y_vec)])
def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b, method):
    pn = np.empty_like(p)
    pn = p.copy()
    if(method == 0):
        for q in range(Nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])
        
        # p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        # p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        # p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        # p[-1, :] = p[-2, :] # dp/dy = 0 at y = 2
            
        p[:, -1] = 0 # p = 0 at x = 2
        # p[0, :] = 0   # p = 0 at y = 0
        p[:, 0] = 0   # p = 0 at x = 0
        # p[-1, :] = 0 # p = 0 at y = 2

        p[0, :] = p[1, :]
        p[-1, :] = p[0, :]
        
    if(method == 1):
        div = np.empty_like(p)
        div[1:-1, 1:-1] = -0.5*dx*( u[2:, 1:-1] - u[:-2, 1:-1] 
                                   +
                                    v[1:-1, 2:] - v[1:-1, :-2])
        for q in range(Nit):
            pn = p.copy()
            p[1:-1, 1:-1] = ( div[1:-1, 1:-1] + pn[2:, 1:-1] + pn[:-2, 1:-1] + pn[1:-1, 2:] + pn[1:-1, :-2] ) / 4

    return p

for t in range(0, Nt):
    u_0 = np.copy(u)
    v_0 = np.copy(v)
    p_0 = np.copy(p)

    # -- Update Pressure -- #
    b = build_up_b(b, rho, dt, u, v, dx, dy)
    p = pressure_poisson(p_0, dx, dy, b, 0)

    # -- Calculate U-Momentum -- # 
    u[1:-1, 1:-1] = ( 
                    u_0[1:-1, 1:-1]  
                    - ( dt / rho ) * (cd_y(p)) \
                    - dt * ( u_0[1:-1, 1:-1] * fd_y(u_0) ) \
                    - dt * ( v_0[1:-1, 1:-1] * fd_x(u_0) ) \
                    + ( nu * dt ) * ( fd_xx(u_0) + fd_yy(u_0) ) 
                    )
    
    # -- Calculate V-Momentum -- # 
    v[1:-1, 1:-1] = ( 
                    v_0[1:-1, 1:-1]  
                    - ( dt / rho ) * (cd_x(p)) \
                    - dt * ( u_0[1:-1, 1:-1] * fd_y(v_0) ) \
                    - dt * ( v_0[1:-1, 1:-1] * fd_x(v_0) ) \
                    + ( nu * dt ) * ( fd_xx(v_0) + fd_yy(v_0) ) 
                    )

    # -- Boundary Conditions -- #
    # Velocity set to zero at the walls

    u[0, :] = u[1, :]
    u[-1, :] = u[0, :] # lidVelocity
    u[:, 0] = 0
    u[:, -1] = 0

    v[0, :] = v[1, :]
    v[-1, :] = v[0, :]
    v[:, 0] = 0
    v[:, -1] = 0

    v[-2, :] = inletVelocity # inlet slit velocity

    # Creating the obstacle
    if(obstacle == 1):        
        r = 15
        ox = int(Ny/1.2)-1
        oy = int(Nx/2)-1
        for x in range(0, len(x_vec)):
            for y in range(0, len(y_vec)):
                if ((x-ox)**2 + (y-oy)**2 <= r**2):
                    u[x, y] = 0
                    v[x, y] = 0
        circle1 = plt.Circle((oy*dx, oy*dy), r*dx, color='b')
    if(obstacle == 2):
        # box
        pass

    if(t%plot_every==0 or t <10):
        print(np.mean((u[1:-1, 2:] - u[1:-1, :-2])/(2*dx) + (v[2:, 1:-1] - v[:-2,1:-1])/(2*dy))) # Divergence

        vel = np.sqrt(np.sqrt(u**2+v**2))
        curl = fd_x(u) + fd_y(v)

        fig = plt.figure(figsize=(3*Ly,2*Lx), dpi=100)
        ax = fig.gca()

        # plotting the pressure field as a contour
        # plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
        # plt.colorbar()
        # # plotting the pressure field outlines
        # plt.contour(X, Y, p, cmap=cm.viridis)  
        # # plotting velocity field
        # plt.quiver(X[::s, ::s], Y[::s, ::s], u[::s, ::s], v[::s, ::s]) 

        # plt.imshow(vel, cmap=cm.inferno)
        # plt.colorbar()

        fig = plt.figure(figsize=(2*Ly,2*Lx), dpi=100)
        plt.streamplot(x_vec, y_vec, u, v, density=[0.9, 2])
        plt.gca().invert_yaxis()

        plt.xlabel('X')
        plt.ylabel('Y')
        # ax.add_patch(circle1)
        plt.pause(.001)
        plt.close()    

plt.show()


