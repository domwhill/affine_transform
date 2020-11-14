"""Implementation of an affine transform
mapping points on a sphere surface to an ellipsoid surface.

Following code from examples:
https://stackoverflow.com/questions/23444060/affine-3d-transformation-in-python
https://stackoverflow.com/questions/9604132/how-to-project-a-point-on-to-a-sphere
"""
import numpy as np
import scipy.linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_ellipsoid_surface_position(phi, theta, a,b,c):
    """ Wolfram: https://mathworld.wolfram.com/Ellipsoid.html

    this should be a,b,c
    """
    r_x = a*np.sin(theta)*np.cos(phi)
    r_y = b*np.sin(theta)*np.sin(phi)
    r_z = c*np.cos(theta)
    return r_x, r_y, r_z


def get_ellipsoid_surface_points(a,b,c):
    """Return evenly distributed points on the surface of an ellipsoid.

    Ellipsoid equation: (x/a)**2 + (y/b)**2 + (z/c)**2 = 1
    """
    phi = np.linspace(0.0,2.0*np.pi, 100)
    theta = np.linspace(-np.pi, np.pi, 100)
    phi_m,theta_m = np.meshgrid(phi,theta)

    r_x, r_y, r_z  = get_ellipsoid_surface_position(phi_m.flatten(), theta_m.flatten(), a, b, c)
    return r_x, r_y, r_z


def get_affine_transform_matrix(a1,a2,a3, b1,b2,b3, x, y):
    """ y = Ax + b find matrix A
    a1,a2,a3 b1,b2,b3 are orthogonal vectors and x, and y are coordinates in frames 1, 2

    W = AV + B
    """
    V = np.column_stack((a1,a2,a3))
    W = np.column_stack((b1,b2,b3))
    A = W.dot(scipy.linalg.inv(V))

    b = y - A.dot(x)
    return A, b


def perform_affine_transfrom(coord, mat, vec):
    """Returns affine transform.

     y = mat.coord + vec
     """
    return mat.dot(coord) + vec


def scatter3d(ax, position, **kwargs):
    ax.scatter(position[:,0], position[:,1], position[:,2])
    return ax


def set_dims(ax):
    ax.set_xlim(-1.0,1.0)
    ax.set_ylim(-1.0,1.0)
    ax.set_zlim(-1.0,1.0)


if __name__ == "__main__":
    # start with unit sphere described by:
    # (x/a)^2 + (y/b)^2 + (z/c)^2 =1   where a=b=c=1
    a,b,c = 1.0,1.0,1.0
    a_final,b_final,c_final = 0.5,1.0,1.0
    # axes of initial unit sphere
    x0 = np.zeros((3)) # (x,y,z) origin in coord stystem 1 (sphere)
    x1 = np.zeros((3)) # (x,y,z) origin in coord system 2 (ellipsoid)

    # Initialise with points on surface of sphere
    r_x,r_y,r_z = get_ellipsoid_surface_points(a,b,c)
    sphere_positions = np.column_stack((r_x, r_y, r_z))

    # axes of sphere
    p1, p2, p3 = np.array([a, 0, 0]).T, np.array([0, b, 0]).T, np.array([0, 0, c]).T
    # axes of ellipsoid
    h1,h2,h3 = p1*(a_final/a), p2*(b_final/b), p3*(c_final/c)

    # Perform coordinate transform onto ellipsoid surface
    A, b = get_affine_transform_matrix(h1, h2, h3, p1, p2, p3, x0, x1)
    ellipsoid_positions = np.array([perform_affine_transfrom(coord, A, b) for coord in sphere_positions])

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    scatter3d(ax1, sphere_positions)
    scatter3d(ax2, ellipsoid_positions)

    for ax in [ax1,ax2]:
        set_dims(ax)

    plt.show()
