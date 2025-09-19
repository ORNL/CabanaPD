# Version 0.4.0

## New Features
 - Added discrete element method (DEM) support, either as stand-alone or as contact forces for peridynamics
 - Added support for multi-material systems (currently only using the same model form)
 - Added mechanics models with plasticity (elastic-perfectly plastic as a reference implementation)
 - Added heat transfer
 - Added optional calculation of peridynamic stress tensor per particle
 - Added interfaces for output of particle fields within subvolumes as a function of time
 - New examples for all new features above

## Bug Fixes and Improvements
- Fixed CMake installs for downstream packages
- Fixed bug for regions and boundary conditions that encompass >50% of the system
- Added Kokkos fences for parallel consistency
- Enabled pre-notches of any orientation
- Enabled custom particle creation during class construction
- Renamed model tags to clarify `Fracture` vs `NoFracture` and `Elastic` vs other mechanics models
- Added output for total system damage
- Fixed output for total system strain energy density (previously only from rank 0)
- Added consistency checks for system size inputs
- Added option to input m-ratio instead of horizon
- Enforced consistent particle counts for boundary conditions and body terms
- Made energy calculation and output optional

## Minimum dependency version updates
 - Cabana 0.7.0 or later is required


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
