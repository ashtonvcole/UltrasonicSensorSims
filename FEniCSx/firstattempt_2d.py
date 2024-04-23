from dolfinx import (
    default_scalar_type,
    mesh
)
from dolfinx.fem import (
    Constant,
    dirichletbc,
    form,
    Function,
    FunctionSpace,
    locate_dofs_geometrical
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_matrix_nest,
    # assemble_vector,
    # assemble_vector_nest,
    # create_matrix,
    # create_matrix_nest,
    create_vector,
    create_vector_nest,
    set_bc
)
from dolfinx.io import VTXWriter

# from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh
# from dolfinx.cpp.mesh import CellType
# from dolfinx.fem import locate_dofs_geometrical, locate_dofs_topological
# from dolfinx.mesh import locate_entities_boundary

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from ufl import (
    div,
    dot,
    dx,
    FiniteElement,
    grad,
    MixedElement,
    TestFunctions,
    TrialFunctions,
    VectorElement
)



# Improvements TODO
# - Generalize dimensions
# - Implement Dirichlet, Neumann, Robin conditions
# - Switch for nonlinear terms (implement nonlinear FEM?)
# - locate_entities for cells for initial conditions? (Interpolation and IO example in docs)



########################
# VARIABLE DEFINITIONS #
########################

# x[3]: cartesian spatial coordinates
# t: time
# rho_p(x, t): unknown SMALL acoustic density perturbation, part of scalar function space V
# rho_bar: density of (roughly) incompressible fluid
# u_p_(x, t): unknown SMALL acoustic velocity perturbation, part of vector function space W
# u_bar_(x): steady-state incompressible velocity field
# c: speed of sound

# d[rho_p]/dt + div(u_p_) + inner(u_bar_, grad(rho_p)) = 0
# d[u_p_]/dt + inner(u_bar_, grad(u_p_)) + c^2 / rho_bar * grad(rho_p) 
#     = -inner(u_bar_, grad(u_bar_))



############################
# USER-PROVIDED PARAMETERS #
############################

# --- USER CONTROLS ---

CASE = 'test'
LOOP = True

# --- DOMAIN ---

xa = 0 # Lower x bound of rectangular mesh
xb = 1 # Upper x bound of rectangular mesh
nx = 100 # Number of elements along x

ya = 0 # Lower y bound of rectangular mesh
yb = 1 # Upper y bound of rectangular mesh
ny = 100 # Number of elements alony y

ta = 0 # Start time
tb = 1 # End time
dt = 0.01 # Time step

# --- FLUID PROPERTIES ---

c = 1 # Speed of sound
rho_bar = 1 # Density
def f_u_bar_(x):
    # Note that x is a tensor of form [[all x's], [all y's], [all z's]]
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 0.1 * ((x[1] * (x[1] - 1)) / -0.25)
    values[1] = 0 * x[0]
    return values

# --- INITIAL CONDITIONS ---

def rho_p_0(x):
    # Note that x is a tensor of form [[all x's], [all y's], [all z's]]
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    for ind in range(1, x.shape[1]):
        if ((x[0][ind] - 0.5) ** 2 + (x[1][ind] - 0.5) ** 2) ** 0.5 < 0.05:\
            values[ind] = x[0][ind] ** 2
        else:
            pass
    return values
def u_p_0_(x):
    # Note that x is a tensor of form [[all x's], [all y's], [all z's]]
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    for ind in range(1, x.shape[1]):
        if ((x[0][ind] - 0.5) ** 2 + (x[1][ind] - 0.5) ** 2) ** 0.5 < 0.05:\
            values[0][ind] = x[0][ind] ** 2
            values[1][ind] = x[1][ind] ** 2
        else:
            pass
    return values



# [!] NO TOUCHY PAST THIS LINE --------------------------------------



#################
# PROBLEM SETUP #
#################

# --- MESH GENERATION ---

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([xa, ya]), np.array([xb, yb])],
    [nx, ny],
    mesh.CellType.triangle
)

# --- BOUNDARY INDICATORS ---

boundary_indicators = [
    lambda x: np.isclose(x[0], xa),
    lambda x: np.isclose(x[0], xb),
    lambda x: np.isclose(x[1], ya),
    lambda x: np.isclose(x[1], yb)
]



##############################
# FINITE ELEMENT FORMULATION #
##############################

# --- ELEMENTS ---

ele_rho_p = FiniteElement('Lagrange', domain.ufl_cell(), degree=1)
ele_u_p_ = VectorElement('Lagrange', domain.ufl_cell(), degree=1, dim=domain.geometry.dim)

# --- FUNCTION SPACES ---

VW_ = FunctionSpace(domain, MixedElement([ele_rho_p, ele_u_p_]))
V, V2VW_ = VW_.sub(0).collapse()
W_, W_2VW_ = VW_.sub(1).collapse()

# --- TRIAL FUNCTIONS ---

(rho_p, u_p_) = TrialFunctions(VW_) # Note plural

# --- TEST FUNCTIONS ---

(v, w_) = TestFunctions(VW_) # Note plural

# --- CONSTANTS ---

k = Constant(domain, default_scalar_type(dt))
c = Constant(domain, default_scalar_type(c))
rho_bar = Constant(domain, default_scalar_type(rho_bar))
u_bar_ = Function(W_)
u_bar_.name = 'u_bar_'
u_bar_.interpolate(f_u_bar_)

# --- SOLUTION VARIABLES ---

t = ta

rho_p_old = Function(V)
rho_p_old.name = 'rho_p_old'
rho_p_old.interpolate(rho_p_0)

u_p_old_ = Function(W_)
u_p_old_.name = 'u_p_old'
u_p_old_.interpolate(u_p_0_)

sol = Function(VW_)
sol.name = 'full_solution'
rho_p_new, u_p_new_ = sol.split()
rho_p_new = rho_p_new.collapse()
rho_p_new.name = 'rho_p'
rho_p_new.interpolate(rho_p_0) # Only to save at time ta
u_p_new_ = u_p_new_.collapse()
u_p_new_.name = 'u_p'
u_p_new_.interpolate(u_p_0_) # Only to save at time ta

# --- BOUNDARY CONDITIONS ---

bcs = []

# Locating them "geometrically"
for indicator in boundary_indicators:
    bcs.append(dirichletbc(
        PETSc.ScalarType(0),
        locate_dofs_geometrical(V, indicator),
        V
    ))
    bcs.append(dirichletbc(
        np.zeros(2, dtype=PETSc.ScalarType),
        locate_dofs_geometrical(W_, indicator),
        W_
    ))

# --- VARIATIONAL FORM ---

# # Bilinear a(u, v), i.e. has solution variable, linear
# a = [
#     # Equation 1
#     rho_p * v * dx +
#     k * div(u_p_) * v * dx +
#     k * dot(u_bar_, grad(rho_p)) * v * dx,
#     # Equation 2
#     dot(u_p_, w_) * dx +
#     dot(k * dot(u_bar_, grad(u_p_)), w_) * dx +
#     dot(k * c * c / rho_bar * grad(rho_p), w_) * dx
# ]
# form_a = form(a)
# 
# # Linear L(v), i.e. no solution variable
# L = [
#     # Equation 1
#     rho_p_old * v * dx,
#     # Equation 2
#     dot(u_p_old_, w_) * dx -
#     dot(k * dot(u_bar_, grad(u_bar_)), w_) * dx
# ]
# form_L = form(L)

# Bilinear a(u, v), i.e. has solution variable, linear
a = (
    # Equation 1
    rho_p * v * dx +
    k * div(u_p_) * v * dx +
    k * dot(u_bar_, grad(rho_p)) * v * dx +
    # Equation 2
    dot(u_p_, w_) * dx +
    dot(k * dot(u_bar_, grad(u_p_)), w_) * dx +
    dot(k * c * c / rho_bar * grad(rho_p), w_) * dx
)
form_a = form(a)

# Linear L(v), i.e. no solution variable
L = (
    # Equation 1
    rho_p_old * v * dx +
    # Equation 2
    dot(u_p_old_, w_) * dx -
    dot(k * dot(u_bar_, grad(u_bar_)), w_) * dx
)
form_L = form(L)

# --- MATRIX EQUATION ---



############
# SOLUTION #
############

A = assemble_matrix(form_a, bcs=bcs)
A.assemble()
b = create_vector(form_L)
apply_lifting(b, [form_a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)

# --- SOLVER ---

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# --- WRITE TO FILE ---

vtx_rho = VTXWriter(domain.comm, f'{CASE}_rho.bp', [rho_p_new])
vtx_rho.write(t)
vtx_u_ = VTXWriter(domain.comm, f'{CASE}_u.bp', [u_p_new_])
vtx_u_.write(t)

# --- SOLUTION LOOP ---

# don't forget to assign new to old before calculating new
# recompile bilinear for updates to k, old
# scatter forward??

while t < tb - 0.1 * dt and LOOP:
    # Update time
    t += dt
    print(f'Time = {t:8.3f} s ({(t - ta) / (tb - ta) * 100:6.2f}%)')
    
    # Move new to old
    rho_p_old.x.array[:] = rho_p_new.x.array
    u_p_old_.x.array[:] = u_p_new_.x.array
    
    # Update b
    apply_lifting(b, [form_a], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    
    # Solve
    solver.solve(b, sol.vector)
    sol.x.scatter_forward()
    
    # Assign to split values for printing
    rho_p_prov, u_p_prov_ = sol.split()
    rho_p_prov = rho_p_prov.collapse()
    u_p_prov_ = u_p_prov_.collapse()
    rho_p_new.x.array[:] = rho_p_prov.x.array
    u_p_new_.x.array[:] = u_p_prov_.x.array
    
    # Write
    vtx_rho.write(t)
    vtx_u_.write(t)

# --- CLOSE FILE ---

vtx_rho.close()
vtx_u_.close() 