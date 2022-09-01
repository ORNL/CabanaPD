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
# Change directory as needed
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

Once configured, build and install CabanaMD with:
```
make -j $BUILD_NUM_THREADS
make install
```
Ensure installation by checking the installed libraries an headers in CBNMD_INSTALL_DIR. If tests are enable you can run the CabanaMD unit test suite with:
```
cd build
ctest
```

## License

CabanaPD is distributed under an [open source 3-clause BSD license](LICENSE).
