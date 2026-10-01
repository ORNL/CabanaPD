#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

#include "polycrystal.hpp"

// Simulate a crack in a polycrystal
void polycrystalExample( const std::string filename )
{
    // ====================================================
    //               Choose Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = { inputs["low_corner"][0],
                                         inputs["low_corner"][1],
                                         inputs["low_corner"][2] };
    std::array<double, 3> high_corner = { inputs["high_corner"][0],
                                          inputs["high_corner"][1],
                                          inputs["high_corner"][2] };

    // ====================================================
    //                Material parameters
    // ====================================================
    double grainRho = inputs["density"][0];

    // intra-grain parameters
    double E_intra = inputs["elastic_modulus"][0];
    double nu_intra = inputs["Poisson's_ratio"][0];
    double G0_intra = inputs["fracture_energy"][0];
    double K_intra = E_intra / ( 3 * ( 1 - 2 * nu_intra ) );

    // inter-grain parameters
    double E_inter = inputs["elastic_modulus"][1];
    double nu_inter = inputs["Poisson's_ratio"][1];
    double G0_inter = inputs["fracture_energy"][1];
    double K_inter = E_inter / ( 3 * ( 1 - 2 * nu_inter ) );

    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                Polycrystal grains
    // ====================================================
    std::array<double, 3> extent = inputs["system_size"];
    double grainSize = inputs["grain_size"];

    // Initialize host storage for outputs from grain generator
    std::vector<std::array<double, 3>> grainPosStd;
    std::array<int, 3> grainGridShape;
    double grainGridCellSize;

    // Generate grains relative to the origin
    getPolycrystalGrains( extent, grainSize, grainPosStd, grainGridShape,
                          grainGridCellSize );

    // initialize grain positions and lookup grid Kokkos host memory
    int numGrains = grainPosStd.size();
    Kokkos::View<double* [3], Kokkos::HostSpace> grainPosHost(
        Kokkos::ViewAllocateWithoutInitializing( "Host grain position" ),
        numGrains );
    Kokkos::View<int***, Kokkos::HostSpace> grainGridHost(
        Kokkos::ViewAllocateWithoutInitializing( "Host grain grid" ),
        grainGridShape[0], grainGridShape[1], grainGridShape[2] );

    // Initialize lookup grid empty (all -1 indices)
    Kokkos::deep_copy( grainGridHost, -1 );

    // Copy and shift grain positions and rebuild lookup grid
    for ( int i = 0; i < numGrains; ++i )
    {
        grainPosHost( i, 0 ) = grainPosStd[i][0] + low_corner[0];
        grainPosHost( i, 1 ) = grainPosStd[i][1] + low_corner[1];
        grainPosHost( i, 2 ) = grainPosStd[i][2] + low_corner[2];

        // Compute grain multi-index and add to the lookup grid
        std::array<int, 3> index = {
            static_cast<int>(
                std::floor( grainPosStd[i][0] / grainGridCellSize ) ),
            static_cast<int>(
                std::floor( grainPosStd[i][1] / grainGridCellSize ) ),
            static_cast<int>(
                std::floor( grainPosStd[i][2] / grainGridCellSize ) ),
        };
        grainGridHost( index[0], index[1], index[2] ) = i;
    }

    // Now copy from host memory to target memory_space
    Kokkos::View<double**, memory_space> grainPos(
        Kokkos::ViewAllocateWithoutInitializing( "Grain positions" ), numGrains,
        3 );
    Kokkos::View<int***, memory_space> grainGrid(
        Kokkos::ViewAllocateWithoutInitializing( "Grain grid" ),
        grainGridShape[0], grainGridShape[1], grainGridShape[2] );
    Kokkos::deep_copy( grainPos, grainPosHost );
    Kokkos::deep_copy( grainGrid, grainGridHost );

    // ====================================================
    //                   Force models
    // ====================================================
    using model_type = CabanaPD::PMB;

    // Grain force models
    CabanaPD::ForceModel force_model_intra( model_type{}, horizon, K_intra,
                                            G0_intra );
    CabanaPD::ForceModel force_model_inter( model_type{}, horizon, K_inter,
                                            G0_inter );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( memory_space{}, model_type{} );
    particles.domain( inputs );
    particles.create( exec_space{} );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    double dy = particles.dx[1];
    CabanaPD::Region<CabanaPD::RectangularPrism> plane1(
        low_corner[0], high_corner[0], low_corner[1] - dy, low_corner[1] + dy,
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> plane2(
        low_corner[0], high_corner[0], high_corner[1] - dy, high_corner[1] + dy,
        low_corner[2], high_corner[2] );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto nofail = particles.sliceNoFail();
    auto type = particles.sliceType();

    // Initialize functor to find closest grains during particle init
    FindClosestGrainFunctor findClosest( grainGrid, grainPos, low_corner,
                                         grainGridShape, grainGridCellSize );

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // No-fail zone
        if ( x( pid, 1 ) <= plane1.low[1] + horizon ||
             x( pid, 1 ) >= plane2.high[1] - horizon )
            nofail( pid ) = 1;

        // Get index of closest grain
        int closestIndex = findClosest( x( pid, 0 ), x( pid, 1 ), x( pid, 2 ) );

        // Set material type and density
        type( pid ) = closestIndex;
        rho( pid ) = grainRho;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::BinaryIndexing indexing;
    auto models = CabanaPD::createMultiForceModel(
        particles, indexing, force_model_intra, force_model_inter );
    CabanaPD::Solver solver( inputs, particles, models );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double v0 = inputs["speed"];
    double midY = ( low_corner[1] + high_corner[1] ) / 2.0;
    f = solver.particles.sliceForce();
    x = solver.particles.sliceReferencePosition();
    auto u = solver.particles.sliceDisplacement();

    // Create symmetric displacement boundary condition
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double t )
    {
        double ypos = x( pid, 1 ) - midY;
        double sign = 0.0;
        if ( ypos > 0 )
        {
            sign = 1.0;
        }
        else
        {
            sign = -1.0;
        }
        u( pid, 1 ) = sign * v0 * t;
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, solver.particles,
                                       true, plane1, plane2 );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    polycrystalExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
