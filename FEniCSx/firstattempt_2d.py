import dolfinx
import numpy as np
import ufl

from dolfinx import default_scalar_type, mesh
from dolfinx.fem import Constant, form, Function, FunctionSpace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile

# from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh
# from dolfinx.cpp.mesh import CellType
# from dolfinx.fem import locate_dofs_geometrical, locate_dofs_topological
# from dolfinx.mesh import locate_entities_boundary

from mpi4py import MPI
from petsc4py import PETSc
from ufl import div, dx, grad, dot, TestFunction, TrialFunction



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
DEBUG = True

# --- DOMAIN ---

xa = 0 # Lower x bound of rectangular mesh
xb = 1 # Upper x bound of rectangular mesh
nx = 10 # Number of elements along x

ya = 0 # Lower y bound of rectangular mesh
yb = 1 # Upper y bound of rectangular mesh
ny = 10 # Number of elements alony y

ta = 0 # Start time
tb = 1 # End time
dt = 0.1 # Time step

# --- FLUID PROPERTIES ---

c = 1 # Speed of sound
rho_bar = 1 # Density
f_u_bar_ = lambda x: np.array([1, 10]) # Note two dimensions, also ha. ha. ha.

# --- INITIAL CONDITIONS ---

rho_p_0 = lambda x: 0
u_p_0_ = lambda x: np.array([0, 0]) # Note two dimensions



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
xdmf = XDMFFile(MPI.COMM_WORLD, f"{CASE}.xdmf", "w")
xdmf.write_mesh(domain)

# --- BOUNDARY INDICATORS ---

boundaries = [
    (1, lambda x: np.isclose(x[0], xa)),
    (2, lambda x: np.isclose(x[0], xb)),
    (3, lambda x: np.isclose(x[1], ya)),
    (4, lambda x: np.isclose(x[1], yb))
]

facet_indices = [] # ID's of facets
facet_markers = [] # Which boundary
fdim = domain.topology.dim

for (marker, indicator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, indicator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

if DEBUG:
    xdmf.write_meshtags(facet_tag, domain.geometry)



##############################
# FINITE ELEMENT FORMULATION #
##############################

# --- FUNCTION SPACES ---

V = FunctionSpace(domain, ("Lagrange", 1))
W_ = FunctionSpace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

# --- TRIAL FUNCTIONS ---

rho_p = ufl.TrialFunction(V)
u_p_ = ufl.TrialFunction(W_)

# --- TEST FUNCTIONS ---

v = ufl.TestFunction(V)
w_ = ufl.TestFunction(W_)

# --- CONSTANTS ---

k = Constant(domain, default_scalar_type(dt))
c = Constant(domain, default_scalar_type(c))
rho_bar = Constant(domain, default_scalar_type(rho_bar))
u_bar_ = Function(W_)
u_bar_.name = 'u_bar_'
u_bar_.interpolate(lambda x: np.vstack((1, -1))) # I don't know why vstack is necessary

if DEBUG:
    xdmf.write_function(u_bar_)
    exit()
    u_bar = u_bar_.sub(0)
    u_bar.name = 'u_bar'
    xdmf.write_function(u_bar)
    v_bar = u_bar_.sub(1)
    v_bar.name = 'v_bar'
    xdmf.write_function(v_bar)
    print(u_bar.x.array)
    def initial_condition(x, a=5):
        return np.exp(-a * (x[0]**2 + x[1]**2))
    uh = Function(V)
    uh.name = "uh"
    uh.interpolate(initial_condition)
    xdmf.write_function(uh, 0)

# --- SOLUTION VARIABLES ---

t = ta

rho_p_old = Function(V)

rho_p_new = Function(V)
rho_p_new.name = 'rho_p'
rho_p_new.interpolate(lambda x: np.vstack((rho_p_0(x),)))

u_p_old_ = Function(W_)

u_p_new_ = Function(W_)
u_p_new_.name = 'u_p (vector)'
u_p_new_.interpolate(lambda x: np.vstack(u_p_0_(x))) # I don't know why vstack is necessary

# --- BOUNDARY CONDITIONS ---

bcs=None

# --- VARIATIONAL FORM ---

# Bilinear a(u, v), i.e. has solution variable, linear
a = [
    # Equation 1
    rho_p * v * dx +
    k * div(u_p_) * v * dx +
    k * dot(u_bar_, grad(rho_p)) * v * dx,
    # Equation 2
    dot(u_p_, w_) * dx +
    dot(k * dot(u_bar_, grad(u_p_)), w_) * dx +
    dot(k * c * c / rho_bar * grad(rho_p), w_) * dx
]

# Linear L(v), i.e. no solution variable
L = [
    # Equation 1
    rho_p_old * v * dx,
    # Equation 2
    dot(u_p_old_, w_) * dx -
    dot(k * dot(u_bar_, grad(u_bar_)), w_) * dx
]

# --- MATRIX EQUATION ---

### A = create_matrix_nest(form(a), bcs=bcs)
### b = create_vector_nest(form(L))



#################
# SOLUTION LOOP #
#################






# don't forget to assign new to old before calculating new
# recompile bilinear for updates to k, old