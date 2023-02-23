#include <cmath>
#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

int main( int argc, char* argv[] )
{

    MPI_Init( &argc, &argv );

    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        // FIXME: change backend at compile time for now.
        using exec_space = Kokkos::DefaultExecutionSpace;
        using memory_space = typename exec_space::memory_space;

        // Plate dimension)
        double radius = 37.0e-3;   // [m]
        double thickness = 2.5e-3; // [m]

        // Domain
        std::array<int, 3> num_cell = { 149, 149, 5 };
        std::array<double, 3> low_corner = { -radius, -radius, -thickness };
        std::array<double, 3> high_corner = { radius, radius, 0.0 };
        double t_final = 2.0e-4;
        double dt = 1.0e-7;
        int output_frequency = 50;

        // Material constants
        // double nu = 1.0 / 4.0; // unitless
        double K = 14.9e+9;   // [Pa]
        double rho0 = 2200.0; // [kg/m^3]
        double G0 = 10.0575;  // [J/m^2]
        double impact_v = -100.0;
        // double ks = 1.0e17;
        double impact_r = 5e-3; // [m]

        // FIXME: set halo width based on delta
        double delta = 0.0015;
        int halo_width = 2;

        // Choose force model type.
        CabanaPD::PMBDamageModel force_model( delta, K, G0 );
        CabanaPD::Inputs inputs( num_cell, low_corner, high_corner, t_final, dt,
                                 output_frequency );
        inputs.read_args( argc, argv );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        using device_type = Kokkos::Device<exec_space, memory_space>;
        auto particles = std::make_shared<CabanaPD::Particles<device_type>>(
            exec_space(), inputs.low_corner, inputs.high_corner,
            inputs.num_cells, halo_width, true );

        // Define particle initialization.
        auto x = particles->slice_x();
        auto v = particles->slice_v();
        auto f = particles->slice_f();
        auto rho = particles->slice_rho();

        CabanaPD::Prenotch<0> prenotch;
        CabanaPD::BoundaryCondition<CabanaPD::AllTag, CabanaPD::ImpactBCTag> bc(
            impact_r, impact_v, dt );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
            for ( int d = 0; d < 3; d++ )
                v( pid, d ) = 0.0;
        };
        particles->update_particles( exec_space{}, init_functor );

        // FIXME: use createSolver to switch backend at runtime.
        auto cabana_pd = CabanaPD::createSolverFracture<device_type>(
            inputs, particles, force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}
