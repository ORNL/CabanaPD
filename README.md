# CabanaPD

Peridynamics with the Cabana library

## Dependencies
CabanaPD has the following dependencies:

|Dependency | Version  | Required | Details|
|---------- | -------  |--------  |------- |
|CMake      | 3.11+    | Yes      | Build system
|Cabana     | f99c7db9 | Yes      | Performance portable particle algorithms
|GTest      | 1.10+    | No       | Unit test framework

Cabana must be built with the following in order to work with CabanaPD:
|Cabana Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.16+   | Yes      | Build system
|MPI        | GPU-Aware if CUDA/HIP enabled | Yes | Message Passing Interface
|Kokkos     | 3.7.0+  | Yes      | Performance portable on-node parallelism
|HDF5       | master  | No       | Particle output
|SILO       | master  | No       | Particle output

The underlying parallel programming models are available on most systems, as is
CMake. Those must be installed first, if not available. Kokkos and Cabana are
available on some systems or can be installed with `spack` (see
https://spack.readthedocs.io/en/latest/getting_started.html):

```
spack install cabana@master+grid+hdf5
```

Alternatively, Kokkos can be built locally, followed by Cabana:
https://github.com/ECP-CoPA/Cabana/wiki/1-Build-Instructions

Build instructions are available for both CPU and GPU. Note that Cabana must be
compiled with MPI and the Grid sub-package.

## Obtaining CabanaPD

Clone the master branch:

```
git clone https://github.com/ORNL/CabanaPD.git
```

## Build and install
### CPU Build

After building Kokkos and Cabana for CPU, the following script will build and install CabanaPD:

```
#Change directory as needed
export CABANA_DIR=$HOME/Cabana/build/install

cd ./CabanaPD
mkdir build
cd build
pwd
cmake \
    -D CMAKE_PREFIX_PATH="$CABANA_DIR" \
    -D CMAKE_INSTALL_PREFIX=install \
    .. ;
make install
```

### CUDA Build

After building Kokkos and Cabana for Cuda:
https://github.com/ECP-CoPA/Cabana/wiki/Build-CUDA

The CUDA build script is identical to that above, but again note that Kokkos
must be compiled with the CUDA backend. 

Note that the same compiler should be used for Kokkos, Cabana, and CabanaPD.

### HIP Build

After building Kokkos and Cabana for HIP:
https://github.com/ECP-CoPA/Cabana/wiki/Build-HIP-and-SYCL#HIP

The HIP build script is identical to that above, except that `hipcc` compiler
must be used:

```
-D CMAKE_CXX_COMPILER=hipcc
```

Note that `hipcc` should be used for Kokkos, Cabana, and CabanaPD.

## Tests

Unit tests can be built by updating the CabanaPD CMake configuration in the
script above with:

```
-D CabanaPD_ENABLE_TESTING=ON
```

GTest is required for CabanaPD unit tests, with build instructions
[here](https://github.com/google/googletest). If tests are enabled, you can run
the CabanaPD unit test suite with:

```
cd CabanaPD/build
ctest
```

## Features

CabanaPD currently includes the following:
  - Force models
    - PD bond-based (pairwise): PMB (prototype microelastic brittle)
    - PD state-based (many-body): LPS (linear peridynamic solid)
    - DEM (contact): normal repulsion, Hertzian, HertzianJKR (Johnson–Kendall–Roberts)
    - Hybrid PD-DEM
    - Multi-material systems can be constructed for any models of the **same category** 
      (bond-based, state-based, contact) above (*Currently 2-material systems only*)
      - Cross-term interactions can be averaged, requiring **identical model** types
 - Mechanical response:
   - Elastic only (no failure)
   - Brittle fracture
   - Elastic-perfectly plastic (*Currently bond-based only*)
 - Thermomechanics (*Currently bond-based only, single material only*)
   - Optional heat transfer
 - Time integration
   - Velocity Verlet
 - Pre-crack creation
 - Particle boundary conditions
   - Body terms which apply to all particles
 - Grid-based particle generation supporting custom geometry
 - Output options
   - Total strain energy density
   - Total damage (if fracture is enabled)
   - Per particle output using HDF5 or SILO
     - Base fields: position (reference or current), displacement, velocity, force, material type
     - Strain energy density, damage
     - Stress
     - LPS fields (if used): weighted volume, dilatation
     - Thermal fields (if used): temperature


## Examples

Once built and installed, CabanaPD `examples/` can be run. Timing and energy
information is output to file and particle output is written to files (if enabled within Cabana) that can be [visualized](#visualizing-with-paraview).
New examples can be created by using any of the current cases as a template.
Most inputs are specified in the example JSON files within the relevant `inputs/` subdirectory; some inputs are set within the `.cpp` files directly.

### Mechanics
Examples which only include mechanics and fracture are within `examples/mechanics`.

 -  The first example is an elastic wave propagating through a cube from an initial Gaussian radial displacement profile from [1]. Assuming the build paths above, the example can be run with:
    ```
    ./CabanaPD/build/install/bin/ElasticWave CabanaPD/examples/mechanics/inputs/elastic_wave.json
    ```

 -  The next example is the Kalthoff-Winkler experiment [2], where an impactor causes crack propagation at an angle from two pre-notches on a steel plate.
    ```
    ./CabanaPD/build/install/bin/KalthoffWinkler CabanaPD/examples/mechanics/inputs/kalthoff_winkler.json
    ```

 -  Another example is crack branching in a pre-notched soda-lime glass plate due to traction loading [3].
    ```
    ./CabanaPD/build/install/bin/CrackBranching CabanaPD/examples/mechanics/inputs/crack_branching.json
    ```

 -  An example with multiple random pre-notches is also available.
    ```
    ./CabanaPD/build/install/bin/RandomCracks CabanaPD/examples/mechanics/inputs/random_cracks.json
    ```

 -  The next example is a fragmenting cylinder due to internal pressure [4]. 
    This problem can either run with PD only or with hybrid PD-DEM contact.
    ```
    ./CabanaPD/build/install/bin/FragmentingCylinder CabanaPD/examples/mechanics/inputs/fragmenting_cylinder.json
    ```

 -  An example highlighting plasticity simulates a tensile test based on an ASTM standard dogbone specimen.
    ```
    ./CabanaPD/build/install/bin/DogboneTensileTest CabanaPD/examples/mechanics/inputs/dogbone_tensile_test.json
    ```

 -  An example demonstrating the peridynamic stress tensor computation simulates a square plate under tension with a circular hole at its center [5].
    ```
    ./CabanaPD/build/install/bin/PlateWithHole CabanaPD/examples/mechanics/inputs/plate_with_hole.json
    ```

 -  An example of multi-material simulation demonstrates crack propagation in a pre-notched plate with a stiff inclusion under traction loading.
    ```
    ./CabanaPD/build/install/bin/CrackInclusion CabanaPD/examples/mechanics/inputs/crack_inclusion.json
    ```

### Powder dynamics
Examples which only include mechanics and fracture are within `examples/dem`.

 -  An example using DEM-only demonstrates powder filling in a container.
    ```
    ./CabanaPD/build/install/bin/PowderFill CabanaPD/examples/mechanics/inputs/powder_fill.json
    ```

### Thermomechanics
Examples which demonstrate temperature-dependent mechanics and fracture are within `examples/thermomechanics`.

 -  The first example is thermoelastic deformation in a homogeneous plate due to linear thermal loading [6].
    ```
    ./CabanaPD/build/install/bin/ThermalDeformation CabanaPD/examples/thermomechanics/thermal_deformation.json
    ```

 -  The second example is crack initiation and propagation in an alumina ceramic plate due to a thermal shock caused by water quenching [7].
    ```
    ./CabanaPD/build/install/bin/ThermalCrack CabanaPD/examples/thermomechanics/thermal_crack.json
    ```

### Thermomechanics with heat transfer
Examples with heat transfer are within `examples/thermomechanics`.

 -  The first example is pseudo-1d heat transfer (no mechanics) in a cube.
    ```
    ./CabanaPD/build/install/bin/ThermalDeformationHeatTransfer CabanaPD/examples/thermomechanics/heat_transfer.json
    ```
    The same example with fully coupled thermomechanics can be run (with a much smaller timestep) using `thermal_deformation_heat_transfer.json`.

 -  The second example is pseudo-1d heat transfer (no mechanics) in a pre-notched cube.
    ```
    ./CabanaPD/build/install/bin/ThermalDeformationHeatTransferPrenotched CabanaPD/examples/thermomechanics/heat_transfer.json
    ```


## Visualizing with Paraview

As mentioned above, the simulation results can be visualized with Paraview or similar applications.  

### How to Install

The installation instructions can be found [here](https://www.paraview.org/download/). Ensure you select the appropriate version based on your operating system.

### Importing Files

Once Paraview is installed, the following simulation output file group should be imported to view the results: `particles_..xmf` for HDF5 or `particles_..silo` for SILO. 

If shown the option to select a reader type, select `XDMF Reader` in the "Open Data With ..." window for HDF5. 

### Viewing Results

Below are some basic guidelines for how to perform the initial steps in order to view and analyze the results. A more in-depth tutorial for Paraview can be found [here](https://docs.paraview.org/en/latest/Tutorials/SelfDirectedTutorial/index.html).

1. Select `Apply` in the lower left-hand Properties window. This will load your simulation data.

2. In the Properties window, under Representation, `Surface` will be selected by default as the geometry representation. Change this to `Point Gaussian`. 

3. Different output fields can be selected within the Coloring menu below Representation. 

4. To control the size of the visualized points, scroll down within the Properties window until the Point Gaussian menu and choose a value for Gaussian Radius.


## References

[1] P. Seleson and D.J. Littlewood, Numerical tools for improved convergence
of meshfree peridynamic discretizations, in Handbook of Nonlocal Continuum
Mechanics for Materials and Structures, G. Voyiadjis, ed., Springer, Cham,
2018.

[2] J.F. Kalthoff and S. Winkler, Failure mode transition at high rates of shear
loading, in Impact Loading and Dynamic Behavior of Materials, C.Y. Chiem, H.-D.
Kunze, and L.W. Meyer, eds., Vol 1, DGM Informationsgesellschaft Verlag (1988)
185-195.

[3] F. Bobaru and G. Zhang, Why do cracks branch? A peridynamic investigation of dynamic brittle fracture, International Journal of Fracture 196 (2015): 59–98.

[4] D.J. Littlewood, M.L. Parks, J.T. Foster, J.A. Mitchell, and P. Diehl, The peridigm meshfree peridynamics code, Journal of Peridynamics and Nonlocal Modeling 6 (2024): 118–148.  

[5] A.S. Fallah, I.N. Giannakeas, R. Mella, M.R. Wenman, Y. Safa, and H. Bahai, On the computational derivation of bond-based peridynamic stress tensor, Journal of Peridynamics and Nonlocal Modeling 2 (2020): 352–378. 

[6] D. He, D. Huang, and D. Jiang, Modeling and studies of fracture in functionally graded materials under thermal shock loading using peridynamics, Theoretical and Applied Fracture Mechanics 111 (2021): 102852.

[7] C.P. Jiang, X.F. Wu, J. Li, F. Song, Y.F. Shao, X.H. Xu, and P. Yan, A study of the mechanism of formation and numerical simulations of crack patterns in ceramics subjected to thermal shock, Acta Materialia 60 (2012): 4540–4550.

## Contributing

We encourage you to contribute to CabanaPD! Please check the
[guidelines](CONTRIBUTING.md) on how to do so.

## Citing CabanaPD

If you use CabanaPD in your work, please cite the [Zenodo release](https://zenodo.org/record/7087781#.Y309w7LMLKI).

## License

CabanaPD is distributed under an [open source 3-clause BSD license](LICENSE).
