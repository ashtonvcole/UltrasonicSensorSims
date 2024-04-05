# Full Cylinder Template

By Ashton Cole

*For use with OpenFOAM v7*

## Steps

1. Inspect, and then edit or replace all of the dictionaries in `system/`, `constant/`, and `0/`.
2. The scripts `Allclean`, `Allrun`, and `AllrunParallel` automate clearing solution data and running the case. Otherwise, refer to step 3.
3. Run `runApplication blockMesh` and `runApplication checkMesh`, then inspect the mesh in ParaView.
	- A new folder `constant/polyMesh/` will be generated with mesh data.
	- Note that `runApplication` is a wrapper script that redirects output to a log file. The command `runParallel` does the same in parallel, using MPI.
4. Run `runApplication icoFoam`.

## OpenFOAM Case Structure

- `case.foam`: An empty file used for opening the case in ParaView.
- `constant/`
	- `transportProperties`: Used to specify the viscosity of the fluid.
- `system/`
	- `blockMeshDict`: A dictionary defining the geometry necessary to generate a block mesh.
	- `controlDict`: A dictionary defining the solver used, time discretization, and result writing preferences, among other things.
	- `decomposeParDict`: A dictionary defining how the domain should be divided to solve in parallel.
	- `fvSchemes`: A dictionary defining numerical schemes used for differential operators.
	- `fvSolution`: A dictionary defining particular solvers and tolerances used.
- `0`
	- `p`: Initial conditions for pressure.
	- `U`: Initial conditions for velocity.
