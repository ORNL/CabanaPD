#include <cmath>
#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

// Simulate a sphere impacting a thin cylindrical plate.
void diskImpactExample( const std::string filename )
{
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    CabanaPD::Inputs inputs( filename );

    double K = inputs["bulk_modulus"];
    double G0 = inputs["fracture_energy"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    // Choose force model type.
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
    model_type force_model( delta, K, G0 );

    // Create particles from mesh.
    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space(), inputs );

    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    double global_mesh_ext = high_corner[0] - low_corner[0];
    auto x = particles->sliceReferencePosition();
    auto create_func = KOKKOS_LAMBDA( const int, const double px[3] )
    {
        auto width = global_mesh_ext / 2.0;
        auto r2 = px[0] * px[0] + px[1] * px[1];
        if ( r2 > width * width )
            return false;
        return true;
    };
    particles->createParticles( exec_space{}, create_func );

    // Define particle initialization.
    auto v = particles->sliceVelocity();
    auto f = particles->sliceForce();
    auto rho = particles->sliceDensity();

    double rho0 = inputs["density"];
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        for ( int d = 0; d < 3; d++ )
            v( pid, d ) = 0.0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    double dx = particles->dx[0];
    double r_c = dx * 2.0;
    CabanaPD::NormalRepulsionModel contact_model( delta, r_c, K );

    // FIXME: use createSolver to switch backend at runtime.
    auto cabana_pd = CabanaPD::createSolverContact<memory_space>(
        inputs, particles, force_model, contact_model );

    double impact_r = inputs["impactor_radius"];
    double impact_v = inputs["impactor_velocity"];
    double impact_x = 0.0;
    double impact_y = 0.0;
    auto body_func = KOKKOS_LAMBDA( const int p, const double t )
    {
        auto z = t * impact_v + impact_r;
        double r = sqrt( ( x( p, 0 ) - impact_x ) * ( x( p, 0 ) - impact_x ) +
                         ( x( p, 1 ) - impact_y ) * ( x( p, 1 ) - impact_y ) +
                         ( x( p, 2 ) - z ) * ( x( p, 2 ) - z ) );
        if ( r < impact_r )
        {
            double fmag = -1.0e17 * ( r - impact_r ) * ( r - impact_r );
            f( p, 0 ) += fmag * x( p, 0 ) / r;
            f( p, 1 ) += fmag * x( p, 1 ) / r;
            f( p, 2 ) += fmag * ( x( p, 2 ) - z ) / r;
        }
    };
    auto body = CabanaPD::createBodyTerm( body_func, true );

    cabana_pd->init( body );
    cabana_pd->run( body );
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::initialize( argc, argv );

    diskImpactExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
