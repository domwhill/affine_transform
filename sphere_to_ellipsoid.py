"""https://stackoverflow.com/questions/23444060/affine-3d-transformation-in-python

See also stereographic projection

https://stackoverflow.com/questions/9604132/how-to-project-a-point-on-to-a-sphere
"""
import sys
sys.path.extend(["/Users/dominichill/software-repo", "/Users/dominichill/software-repo/src"])
import logging
import numpy as np
import numpy
import scipy.linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def get_surface_musp(nodes, reduced_scattering_coefficients):
    """Get the surface reduced scattering coefficient.
    Args:
        nodes ():
        reduced_scattering_coefficients ():

    Returns:

    """
    return np.average(reduced_scattering_coefficients[nodes[:, -1] == np.max(nodes[:, -1])])


def get_ellipsoid_eq(x,y,z, a = 1,b = 1, c = 1):
    return (x/a)**2 + (y/b)**2 + (z/c)**2

def get_ellipsoid_surface_position(phi, theta, a,b,c):
    """ Woflfram: this should be a,b,c"""
    r_x = a*np.sin(theta)*np.cos(phi)
    r_y = b*np.sin(theta)*np.sin(phi)
    r_z = c*np.cos(theta)
    return r_x, r_y, r_z

def get_ellipsoid_surface_points(a,b,c):

    phi = np.linspace(0.0,2.0*np.pi, 100)
    theta = np.linspace(-np.pi, np.pi, 100)
    phi_m,theta_m = np.meshgrid(phi,theta)

    r_x, r_y, r_z  = get_ellipsoid_surface_position(phi_m.flatten(), theta_m.flatten(), a, b, c)
    return r_x, r_y, r_z


def get_affine_transform_matrix(a1,a2,a3, b1,b2,b3, x, y):
    """ y = Ax + b find matrix A
    a1,a2,a3 b1,b2,b3 are orthoganl vectors and x, and y are coordinates in frames 1, 2

    W = AV + B

    """

    V = np.column_stack((a1,a2,a3))
    W = np.column_stack((b1,b2,b3))
    A = W.dot(scipy.linalg.inv(V))

    b = y - A.dot(x)
    return A, b

def perform_affine_transfrom(coord, mat, vec):
    """Returns affine transform

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
    return ax


if __name__ == "__main__":

    # Take the cross product of the vertices vectors
    # going from plane -> head
    # start with unit sphere described by:
    # (x/a)^2 + (y/b)^2 + (z/c)^2 =1   where a=b=c=1
    a,b,c = 1.0,1.0,1.0
    a_final,b_final,c_final = 0.5,1.0,1.0
    # axes of initial unit sphere
    p1, p2, p3 = np.array([a,0,0]).T, np.array([0,b,0]).T, np.array([0,0,c]).T

    x = np.zeros((3)) # origin in coord stystem 1
    y = np.zeros((3)) # origin in coord system 2
    # axes of ellipsoid
    h1,h2,h3 = p1*(a_final/a), p2*(b_final/b), p3*(c_final/c)
    A, b = get_affine_transform_matrix(h1, h2, h3, p1, p2, p3, x, y)
    # positions on a unit sphere
    r_x,r_y,r_z = get_ellipsoid_surface_points(a,b,c)
    s_positions = np.column_stack((r_x,r_y,r_z))
    # transform onto ellipsoid
    transformed_coordinates = np.array([perform_affine_transfrom(coord, A, b) for coord in s_positions])

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    scatter3d(ax1, s_positions)
    scatter3d(ax2, transformed_coordinates)

    for ax in [ax1,ax2]:
        ax = set_dims(ax)

    plt.show()