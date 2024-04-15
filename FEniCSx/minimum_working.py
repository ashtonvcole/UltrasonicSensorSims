# Minimum Working Example

import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc



def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])],
                               [50, 50], mesh.CellType.triangle)
xdmf = io.XDMFFile(domain.comm, "minimum_working.xdmf", "w")
xdmf.write_mesh(domain)

V = fem.FunctionSpace(domain, ("Lagrange", 1))

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, 0)