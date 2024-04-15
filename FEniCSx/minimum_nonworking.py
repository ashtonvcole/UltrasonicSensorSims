# Minimum Nonworking Example

import dolfinx
import numpy as np
import ufl

from dolfinx import default_scalar_type, mesh
from dolfinx.fem import Constant, form, Function, FunctionSpace, VectorFunctionSpace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile

# from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh
# from dolfinx.cpp.mesh import CellType
# from dolfinx.fem import locate_dofs_geometrical, locate_dofs_topological
# from dolfinx.mesh import locate_entities_boundary

from mpi4py import MPI
from petsc4py import PETSc
from ufl import div, dx, grad, dot, TestFunction, TrialFunction, VectorElement


CAUSE_PROBLEMS = False
TRY_FIX_1 = False
TRY_FIX_2 = True


def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([1, 1])],
    [10, 10],
    mesh.CellType.triangle
)
xdmf = XDMFFile(MPI.COMM_WORLD, "minimum_nonworking.xdmf", "w")
xdmf.write_mesh(domain)

V = FunctionSpace(domain, ("Lagrange", 1))

uh = Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, 0)

# Print out values to check
print(uh.x.array)
print(len(uh.x.array))



if CAUSE_PROBLEMS:
    # W_ = FunctionSpace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    W_ = FunctionSpace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    # W_ = VectorFunctionSpace(domain, ('Lagrange', 1), domain.geometry.dim)

    u_bar_ = Function(W_)
    u_bar_.name = 'u_bar_'
    # u_bar_.interpolate(lambda x: np.vstack((1, -1))) # I don't know why vstack is necessary
    u_bar_.interpolate(lambda x: np.array(((1,), (-1,)))) # Still doesn't work
    xdmf.write_function(u_bar_, 0)
    
    # Print out values to check
    print(u_bar_.x.array)
    print(len(u_bar_.x.array))



if TRY_FIX_1:
    w_ele_ = VectorElement('Lagrange', domain.ufl_cell(), 2) # Should be dimension of mesh
    W_ = FunctionSpace(domain, w_ele_)
    
    u_bar_ = Function(W_)
    u_bar_.name = 'u_bar_'
    u_bar_.interpolate(lambda x: np.vstack((1, -1))) # I don't know why vstack is necessary
    xdmf.write_function(u_bar_, 0)
    
    # Print out values to check
    print(u_bar_.x.array)
    print(len(u_bar_.x.array))



if TRY_FIX_2:
    v_cg2 = VectorElement("Lagrange", domain.ufl_cell(), 2, dim=2)
    V = FunctionSpace(domain, v_cg2)
    
    def u_exact(x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = [1,]
        values[1] = [-1,]
        return values
    
    u_ = Function(V)
    u_.name = "u_"
    u_.interpolate(u_exact)
    xdmf.write_function(u_, 0)
    
    # Print out values to check
    print(u_.x.array)
    print(len(u_.x.array))