import numpy as np

from dolfinx import default_scalar_type, mesh
from dolfinx.fem import Constant, form, Function, FunctionSpace, VectorFunctionSpace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector, apply_lifting, set_bc
from dolfinx.io import VTXWriter

from mpi4py import MPI
from petsc4py import PETSc
from ufl import div, dx, grad, dot, FiniteElement, MixedElement, split, TestFunction, TrialFunction, VectorElement



# Values to assign

def p_0(x):
    return (x[0] + x[1])

def u_0(x):
    value = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    value[0] = x[0]
    value[1] = x[1]
    return value



# Create mesh

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                               [5, 5], mesh.CellType.triangle)



# Create function space

ele_p = FiniteElement('Lagrange', domain.ufl_cell(), degree=1)
ele_u_ = VectorElement('Lagrange', domain.ufl_cell(), degree=1, dim=domain.geometry.dim)
MS = FunctionSpace(domain, MixedElement([ele_p, ele_u_]))
UU = Function(MS) # Combined solution variable (p, u_ = (u, v))



# Split and assign values

p, u_ = UU.split() # Not to be confused with split(UU)???
p = p.collapse() # Does something useful
p.name = 'p'
p.interpolate(p_0)
u_ = u_.collapse() # Does something useful
u_.name = 'u'
u_.interpolate(u_0)
print(np.reshape(np.array(u_.x.array), (36, 2)))




# Write to file, now with VTX

vtx = VTXWriter(domain.comm, "weird.bp", [p, u_])
vtx.write(0)
vtx.close()

vtx = VTXWriter(domain.comm, "works.bp", [u_])
vtx.write(0)
vtx.close()



# Not using mixed element

V = FunctionSpace(domain, ele_p)
p = Function(V)
p.name = 'p'
p.interpolate(p_0)

W_ = FunctionSpace(domain, ele_u_)
u_ = Function(W_)
u_.name = "u_"
u_.interpolate(u_0)

vtx = VTXWriter(domain.comm, "third.bp", [p, u_])
vtx.write(0)
vtx.close()