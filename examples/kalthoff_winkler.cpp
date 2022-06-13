#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD_Boundary.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Solver.hpp>

int main( int argc, char* argv[] )
{

    MPI_Init( &argc, &argv );

    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        // FIXME: change backend at compile time for now.
        using memory_space = Kokkos::HostSpace;
        using exec_space = Kokkos::Serial;

        // Plate dimension)
        double height = 0.1;      // [m] (100 mm)
        double width = 0.2;       // [m] (200 mm)
        double thickness = 0.009; // [m] (  9 mm)

        // Domain
        std::array<int, 3> num_cell = { 41, 81, 4 };
        std::array<double, 3> low_corner = { -0.5 * height, -0.5 * width,
                                             -0.5 * thickness };
        std::array<double, 3> high_corner = { 0.5 * height, 0.5 * width,
                                              0.5 * thickness };
        double t_final = 140e-6;
        double dt = 0.2e-6;

        // Material constants
        double E = 191e+9;                     // [Pa]
        double nu = 1 / 3;                     // unitless
        double K = E / ( 3 * ( 1 - 2 * nu ) ); // [Pa]
        double rho0 = 8000;                    // [kg/m^3]
        double G0 = 42408;                     // [J/m^2]

        double v0 = 16;              // [m/sec] (Half impactor's velocity)
        double L_prenotch = 0.05;    // [m] (50 mm)
        double y_prenotch1 = -0.025; // [m] (-25 mm)
        double y_prenotch2 = 0.025;  // [m] ( 25 mm)
        Kokkos::Array<double, 3> p01 = { low_corner[0], y_prenotch1,
                                         low_corner[2] };
        Kokkos::Array<double, 3> p02 = { low_corner[0], y_prenotch2,
                                         low_corner[2] };
        Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
        Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
        Kokkos::Array<Kokkos::Array<double, 3>, 2> notch_positions = { p01,
                                                                       p02 };
        CabanaPD::Prenotch<2> prenotch( v1, v2, notch_positions );

        // FIXME: set halo width based on delta
        double delta = 0.0038;
        int halo_width = 2;

        CabanaPD::InputsFracture inputs( num_cell, low_corner, high_corner, K,
                                         delta, t_final, dt, G0 );
        inputs.read_args( argc, argv );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<CabanaPD::Particles<memory_space>>(
            exec_space(), inputs.low_corner, inputs.high_corner,
            inputs.num_cells, halo_width );

        // Define particle initialization.
        auto x = particles->slice_x();
        auto v = particles->slice_v();
        auto f = particles->slice_f();
        auto rho = particles->slice_rho();

        double dx = particles->dx;

        double x_bc = -0.5 * height;
        CabanaPD::RegionBoundary plane(
            x_bc - dx, x_bc + dx * 1.25, y_prenotch1 - dx * 0.25,
            y_prenotch2 + dx * 0.25, -thickness, thickness );

        auto bc = createBoundaryCondition( exec_space{}, *particles, plane,
                                           CabanaPD::ForceBCTag{} );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
            // Set the x velocity inside the pre-notches.
            if ( x( pid, 1 ) > y_prenotch1 && x( pid, 1 ) < y_prenotch2 &&
                 x( pid, 0 ) < -0.5 * height + dx )
                v( pid, 0 ) = v0;
        };
        particles->update_particles( exec_space{}, init_functor );

        // Choose force model type.
        CabanaPD::PMBDamageModel force_model( inputs.K, inputs.delta,
                                              inputs.G0 );

        // FIXME: use createSolver to switch backend at runtime.
        auto cabana_pd = std::make_shared<CabanaPD::SolverFracture<
            Kokkos::Device<exec_space, memory_space>, CabanaPD::PMBDamageModel,
            decltype( bc ), decltype( prenotch )>>( inputs, particles,
                                                    force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}
