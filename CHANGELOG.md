
# Version 0.3.0

## New Features
 - Added thermomechanics and associated example problems
 - Custom particle generation with user-functors 
 - Added body terms as a special case of boundary conditions applied to the entire system
 - New fragmenting cylinder example

## Bug Fixes and Improvements
- Fixed CMake version information
- Improve generality of boundary conditions
- Added options for linear profile output for particle properties
- Added timestep selection functionality
- Improved solver generality (boundary conditions for non-fracture, fracture without pre-cracks)

## Performance
- Improved performance by reducing kernel launches in halo search

## Minimum dependency version updates
 - Cabana `master` is still required (post-release 0.6.1)


# Version 0.2.0

## New Features
 - New example for crack branching
 - Generalization of boundary conditions (including for crack branching)
 - Addition of "no-fail" zone option (used for crack branching)
 - Added Cabana HDF5 particle output

## Bug Fixes and Improvements
- Incorrect global particle count corrected (using `unsigned long long`)
- Added continuous integration
- Updated build for standard library/executable installation
- Improved performance timers
- Removed uses of `Cabana::Impl` functions

## Performance
- Added `Cabana::Gather` with persistent communication buffers for substantial scalability improvement
- Updates for force kernels with damage to reduce `if` statement branching

## Minimum dependency version updates
 - Cabana `master` is still required: now `31ba70d9` or newer (post-release 0.5.0)


# Version 0.1.0

## New Features
- Cabana particle data on regular grids 
- Multi-node Cabana particle migration
- Cabana Verlet neighbor list
- Prototype microelastic brittle (PMB) and linear peridynamic solid (LPS) force models
- Velocity Verlet time integration
- Pre-crack creation
- Particle boundary conditions
- Elastic wave propagation and Kalthoff-Winkler fracture examples
