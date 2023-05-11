# CabanaPD

Peridynamics with the Cabana library

## Dependencies
CabanaPD has the following dependencies:

|Dependency | Version  | Required | Details|
|---------- | -------  |--------  |------- |
|CMake      | 3.11+    | Yes      | Build system
|Cabana     | 10cc6758 | Yes      | Performance portable particle algorithms
|GTest      | 1.10+    | No       | Unit test framework

Cabana must be built with the following in order to work with CabanaPD:
|Cabana Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.16+   | Yes      | Build system
|MPI        | GPU-Aware if CUDA/HIP enabled | Yes | Message Passing Interface
|Kokkos     | 3.6.0+  | Yes      | Performance portable on-node parallelism
|SILO       | master  | Yes      | Particle output

The underlying parallel programming models are available on most systems, as is
CMake. Those must be installed first, if not available. Kokkos and Cabana are
available on some systems or can be installed with `spack` (see
https://spack.readthedocs.io/en/latest/getting_started.html):

```
spack install cabana@master+cajita+silo
```

Alternatively, Kokkos can be built locally, followed by Cabana:
https://github.com/ECP-copa/Cabana/wiki/Build-Instructions

Build instructions are available for both CPU and GPU. Note that Cabana must be
compiled with Cajita and MPI.

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
https://github.com/ECP-copa/Cabana/wiki/CUDA-Build

The CUDA build script is identical to that above, but again note that Kokkos
must be compiled with the CUDA backend. 

Note that the same compiler should be used for Kokkos, Cabana, and CabanaPD.

### HIP Build

After building Kokkos and Cabana for HIP:
https://github.com/ECP-copa/Cabana/wiki/HIP-and-SYCL-Build#HIP

The HIP build script is identical to that above, except that `hipcc` compiler
must be used:

```
-D CMAKE_CXX_COMPILER=hipcc
```

Note that `hipcc` should be used for Kokkos, Cabana, and CabanaPD.

## Test

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

## Examples

Once built and installed, CabanaPD examples can be run. Timing and energy
information is output to file and particle output is written to files that can
be visualized with Paraview and similar applications. The first example is an
elastic wave propagating through a cube from an initial Gaussian radial
displacement profile from [1]. Assuming the build paths above, the example can
be run with:

```
./CabanaPD/build/install/bin/ElasticWave
```

The second example is the Kalthoff-Winkler experiment [2], where an impactor
causes crack propagation at an angle from two pre-notches on a steel plate. The
example can be run with:

```
./CabanaPD/build/install/bin/KalthoffWinkler
```

New examples can be created by using the existing `KalthoffWinkler` as a
template to simulate other fracture problems. All inputs are currently
specified in `examples/kalthoff_winkler.cpp`

## References

[1] P. Seleson and D.J. Littlewood, Numerical tools for improved convergence
of meshfree peridynamic discretizations, in Handbook of Nonlocal Continuum
Mechanics for Materials and Structures, G. Voyiadjis, ed., Springer, Cham,
2018.

[2] J.F. Kalthoff and S. Winkler, Failure mode transition at high rates of shear
loading, in Impact Loading and Dynamic Behavior of Materials, C.Y. Chiem, H.-D.
Kunze, and L.W. Meyer, eds., Vol 1, DGM Informationsgesellschaft Verlag (1988)
185-195.

## Citing CabanaPD

If you use CabanaPD in your work, please cite the [Zenodo release](https://zenodo.org/record/7087781#.Y309w7LMLKI).

## License

CabanaPD is distributed under an [open source 3-clause BSD license](LICENSE).
