####################################################
# THERMAL EXPANSION - LINEAR ELASTICITY
####################################################


# Importing required libraries
'''
log: Logging utility to display runtime messages
default_scalar_type: Provides default data type for scalars
'''
from dolfinx import log, default_scalar_type
'''
NonlinearProblem: A class to define nonlinear variational problems for use with PETSc-based solvers
'''
from dolfinx.fem.petsc import NonlinearProblem
'''
NewtonSolver: A Newton-Raphson solver from the PETSc nonlinear solvers module
'''
from dolfinx.nls.petsc import NewtonSolver
'''
pyvista: A 3D plotting and mesh visualization library built on top of VTK
'''
import pyvista
'''
numpy: Python library for numerical computing
'''
import numpy as np
'''
ufl: The Unified Form Language used in FEniCSx to define variational forms (e.g., weak forms of PDEs). It’s the symbolic language for mathematical expressions.
'''
import ufl
'''
sym(A): Returns the symmetric part of a tensor A, i.e., $\frac{1}{2}(A + A^T)$
grad(u): Gradient of a scalar or vector function u
Identity(n): Identity tensor of order n × n
tr(A): Trace of a tensor A, i.e., sum of its diagonal elements
inner(A, B): Inner product of two tensors (like dot(A, B) but generalized to tensors)
'''
from ufl import sym, grad, Identity, tr, inner
'''
MPI: The Message Passing Interface used to enable parallel computations across processes. This lets FEniCSx scale to many CPUs for large simulations.
'''
from mpi4py import MPI
'''
fem: For finite element-related operations like function spaces, forms, boundary conditions, etc.
mesh: For creating and manipulating computational meshes
plot: For creating visualizations of meshes and solutions
'''
from dolfinx import fem, mesh, plot

# GEOMETRY AND MESH GENERATION
L = 20.0    # Length of beam = 20, width = height = 1
# Generating hexahedral elements structured 3D box mesh
domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, 1, 1]], [20, 5, 5], mesh.CellType.hexahedron)
# Creating functionspace to which function, test function and trial function belong
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

# DEFINING BOUNDARY MARKERS
def left(x): return np.isclose(x[0], 0)     # left boundary
def right(x): return np.isclose(x[0], L)    # right boundary

fdim = domain.topology.dim - 1              # Topological dimension for the surfaces of the mesh. 2 for this case
# Finding boundary facets belonging to 'left' and 'right' boundary markers
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

# setting markers for the left (1) boundary facet and right (2) boundary facet
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
# Sorting the marked facets. It returns the indices that would sort the array. FEniCS requires the facet indices to be sorted when creating meshtags
sorted_facets = np.argsort(marked_facets)
# Creating meshtags to apply boundary conditions later
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# APPLYING FIXED BOUNDARY CONDITION
# Creating zero displacement vector to fix a boundary
u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
# locating degrees of freedom corresponding to facet tag = 1, i.e., left
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
# Applying fixed boundary condition on left boundary in the function space V that was created earlier
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

# CREATING CONSTANT BODY FORCE VECTOR
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# INITIALIZING TEMPERATURE FIELD AT 0
T = fem.Constant(domain, default_scalar_type(0.0))

# CREATING TEST AND TRIAL FUNCTION
v = ufl.TestFunction(V)
u = fem.Function(V)

# KINEMATICS

# Strain (Please refer to Elasticity by Timoshenko to check how thermal strain are implemented)
def epsilon(u):
    return sym(grad(u)) - alpha*Identity(len(u))*T

# Stress
def sigma(u):
    return 2.0*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(len(u))

# MATERIAL PARAMETERS FOR STEEL
E = default_scalar_type(210.0e9)                                    # Young's modulus
nu = default_scalar_type(0.3)                                       # Poisson's ratio
alpha = default_scalar_type(12e-6)                                  # Coefficient of thermal expansion
mu = fem.Constant(domain, E / (2 * (1 + nu)))                       # Shear modulus
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))    # Lame's parameter


# VARIATIONAL FORM
# specifying that all integrals dx (Volume), dx (surface) should use 4th-order Gaussian quadrature
metadata = {"quadrature_degree": 4}
# Defining surface integration measure ds and volume integration measure dx
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Writing residual form for strain energy - work done by body forces
F_form = inner(sigma(u),grad(v)) * dx - ufl.inner(v, B) * dx

# NONLINEAR PROBLEM AND SOLVER
# defining nonlinear problem (although linear solver should suffice for this particular problem)
problem = NonlinearProblem(F_form, u, bcs)
# defining solver
solver = NewtonSolver(domain.comm, problem)
# defining absolute and relative tolerances
solver.atol = 1e-8
solver.rtol = 1e-8
# Setting convergence criterion for Newton solver to be incremental. This means that it will converge when norm of the update step becomes very small
solver.convergence_criterion = "incremental"

# VISUALIZATION
# Starting a virtual framebuffer (Xvfb), which is needed for offscreen rendering
pyvista.start_xvfb()
# Creating a new 3D rendering scene
plotter = pyvista.Plotter()
# Preparing plotter to save frames as a GIF
plotter.open_gif("deformation.gif", fps=3)

# Converting the FE mesh into a format compatible with PyVista
topology, cells, geometry = plot.vtk_mesh(u.function_space)
# Constructing the 3D mesh that PyVista can render
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# Initializing an array to store displacement vectors
values = np.zeros((geometry.shape[0], 3))
# values[i] = [u_x, u_y, u_z] at vertex i
values[:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
# Adding displacement field 'u' to PyVista mesh as a vector fueld
function_grid["u"] = values
function_grid.set_active_vectors("u")
# Warping the mesh using displacement vectors
warped = function_grid.warp_by_vector("u", factor=1)
warped.set_active_vectors("u")
# Adding the warped mesh to the plot
actor = plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, 10])

# Creating a scalar-valued function space on same mesh
Vs = fem.functionspace(domain, ("Lagrange", 2))
magnitude = fem.Function(Vs)    # a function that will hold norm of u
# Constructing an expression for the magnitude of displacement (|u|)
us = fem.Expression(ufl.sqrt(sum([u[i]**2 for i in range(len(u))])), Vs.element.interpolation_points())
# Projecting the magnitude expression into the finite element space Vs
magnitude.interpolate(us)
# Adding the displacement magnitude field to the warped mesh as a scalar field
warped["mag"] = magnitude.x.array

# TIME STEPPING LOOP
#Sets the logging level to INFO, so FEniCSx prints useful solver details (e.g., residuals, convergence info) during the simulation
log.set_log_level(log.LogLevel.INFO)
tval0 = 200 # Initial temperature
# Storing a handle to the text label in PyVista scene so that it can be updated at each time step
text_actor = None  

# Initializing loop to raise temperature starting from 0 to 2000 in incremental steps of 200
for n in range(1, 11):
    # Incrementing temperature
    T.value = n * tval0

    ### SOLVER STEP ###
    num_its, converged = solver.solve(u)

    # Ensuring solve succeded
    assert converged
    u.x.scatter_forward()

    # Printing time step and number of iteration to keep track while solving
    print(f"Time step {n}, Number of iterations {num_its}, Temperature {T.value}")

    # Updating the vector field "u" with new displacement from this time step.
    function_grid["u"][:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
    # Recomputing the displacement magnitude by interpolating us
    magnitude.interpolate(us)

    # Activating "mag" as the coloring scalar field
    warped.set_active_scalars("mag")
    # Warping the mesh again using new displacement
    warped_n = function_grid.warp_by_vector(factor=1)
    warped.points[:, :] = warped_n.points
    # Updating the color field on the warped mesh
    warped.point_data["mag"][:] = magnitude.x.array

    # Remove the previous temperature label, if it exists
    if text_actor is not None:
        plotter.remove_actor(text_actor)

    # Add updated temperature label and store the actor
    text_actor = plotter.add_text(
        f"T = {T.value:.0f} °C",
        position="upper_edge",
        font_size=20,
        color="black",
        shadow=True
    )

    # Updating the colorbar range
    plotter.update_scalar_bar_range([0, 0.5])
    # Writing current frame to the animated GIF
    plotter.write_frame()

# Closing the plotter to finalize the animation file
plotter.close()
