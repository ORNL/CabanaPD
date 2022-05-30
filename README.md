# CabanaPD

Peridynamics with the Cabana library

## Dependencies
CabanaPD has the following dependencies:

|Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.11+   | Yes      | Build system
|MPI        | GPU-Aware if CUDA/HIP enabled | Yes | Message Passing Interface
|Kokkos     | 3.2.0+  | Yes      | Performance portable on-node parallelism
|Cabana     | 0.4.0+  | Yes      | Performance portable particle algorithms
|CUDA       | 10+     | No       | Programming model for NVIDIA GPUs
|HIP        | 4.2+    | No       | Programming model for AMD GPUs

The underlying parallel programming models are available on most systems, as is
CMake. Those must be installed first, if not available. Kokkos and Cabana are
available on some systems or can be installed with `spack` (see
https://spack.readthedocs.io/en/latest/getting_started.html):

```
spack install cabana+mpi
```

Alternatively, Kokkos can be built locally, followed by Cabana:
https://github.com/ECP-copa/Cabana/wiki/Build-Instructions

Build instructions are available for both CPU and GPU. Note that Cabana must be
compiled with Cajita and MPI.

## Obtaining CabanaPD

Clone the master branch:

```
git clone https://code.ornl.gov/5t2/CabanaPD.git
```

## CPU Build
After building Kokkos and Cabana for CPU:
```
#Change directory as needed
export CABANA_DIR=$HOME/Cabana

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

## CUDA Build
After building Kokkos and Cabana for Cuda:
https://github.com/ECP-copa/Cabana/wiki/CUDA-Build

The CUDA build script is identical to that above, but again note that Kokkos
must be compiled with the Cuda backend. Older versions of Kokkos require a
compiler wrapper to be passed explicitly for Cuda:
`-DCMAKE_CXX_COMPILER=/path/to/nvcc_wrapper`

## HIP Build
After building Kokkos and Cabana for HIP:
https://github.com/ECP-copa/Cabana/wiki/HIP-and-SYCL-Build#HIP

The HIP build script is identical to that above, except that `hipcc` compiler
must be used: `-DCMAKE_CXX_COMPILER=hipcc`. Again note that Kokkos must be
compiled with the HIP backend.

## Build, Test, and Install

Once configured, build and install CabanaPD with:
```
make install
```

If tests are enabled you can run the CabanaPD unit test suite with:
```
cd CabanaPD/build
ctest
```

## Examples

Once built and installed, CabanaPD examples can be run. Timing and energy
information is output to file and particle output (if enabled in Cabana) is
written to files that can be visualized with Paraview and similar applications.
The first is an elastic wave propagating through a cube from an initial
Gaussian displacement profile from [1]. Assuming the build paths above:

```
./CabanaPD/build/examples/ElasticWave
```

The second is the Kalthoff Winkler experiment [2], where an impactor causes
crack propagation at an angle from two pronotches.

```
./CabanaPD/build/examples/KalthoffWinkler
```

New examples can be created by using the existing `KalthoffWinkler` as a
template to simulate other fracture problems. All inputs are currently
specified in `examples/kalthoff_winkler.cpp`

## References

[1] P. Seleson and D. J. Littlewood, Numerical tools for improved convergence
of meshfree peri-dynamic discretizations, in Handbook of Nonlocal Continuum
Mechanics for Materials andStructures, George Voyiadjis, ed., Springer, Cham,
2018.

[2] J.F. Kalthoff, S. Winkler, Failure mode transition at high rates of shear
loading, ImpactLoading and Dynamic Behavior of Materials, C.Y. Chiem, H.-D.
Kunze, and L.W. Meyer, eds.,Vol 1, DGM Informationsgesellschaft Verlag (1988)
185-195.

## License

CabanaPD is distributed under an [open source 3-clause BSD license](LICENSE).
