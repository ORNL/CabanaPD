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

        // Plate dimension) (m)
        double height = 0.04;
        double width = 0.1;
        double thickness = 0.002;

        // Domain
        std::array<int, 3> num_cell = { 40, 100, 4 }; // 300 x 120 x 6
        double low_x = -0.5 * height;
        double low_y = -0.5 * width;
        double low_z = -0.5 * thickness;
        double high_x = 0.5 * height;
        double high_y = 0.5 * width;
        double high_z = 0.5 * thickness;
        std::array<double, 3> low_corner = { low_x, low_y, low_z };
        std::array<double, 3> high_corner = { high_x, high_y, high_z };
        double t_final = 43e-6;
        double dt = 6.7e-8;
        double output_frequency = 1;

        // Material constants
        double E = 72e+9;                      // [Pa]
        double nu = 1 / 3;                     // unitless
        double K = E / ( 3 * ( 1 - 2 * nu ) ); // [Pa]
        double rho0 = 2440;                    // [kg/m^3]
        double G0 = 3.8;                       // [J/m^2]

        double L_prenotch = width / 2.0;
        double x_prenotch1 = 0.0;
        Kokkos::Array<double, 3> p01 = { x_prenotch1, low_y, low_z };
        Kokkos::Array<double, 3> v1 = { 0, L_prenotch, 0 };
        Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
        Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_positions = { p01 };
        CabanaPD::Prenotch<1> prenotch( v1, v2, notch_positions );

        // FIXME: set halo width based on delta
        double delta = 0.0038;
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cell[0] ) );
        int halo_width = m + 1;

        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
        model_type force_model( delta, K, G0 );
        CabanaPD::Inputs inputs( num_cell, low_corner, high_corner, t_final, dt,
                                 output_frequency );
        inputs.read_args( argc, argv );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        using device_type = Kokkos::Device<exec_space, memory_space>;
        auto particles = std::make_shared<
            CabanaPD::Particles<device_type, typename model_type::base_model>>(
            exec_space(), inputs.low_corner, inputs.high_corner,
            inputs.num_cells, halo_width );

        // Define particle initialization.
        auto x = particles->slice_x();
        auto v = particles->slice_v();
        auto f = particles->slice_f();
        auto rho = particles->slice_rho();

        // Relying on uniform volume here.
        double dx = particles->dx;
        double b0 = 2e6 / dx; // Pa

        CabanaPD::RegionBoundary plane1( low_x - dx, low_x + dx, -width, width,
                                         -thickness, thickness );
        CabanaPD::RegionBoundary plane2( high_x - dx, high_x + dx, -width,
                                         width, -thickness, thickness );
        std::vector<CabanaPD::RegionBoundary> planes = { plane1, plane2 };
        auto bc =
            createBoundaryCondition( CabanaPD::ForceUpdateBCTag{}, exec_space{},
                                     *particles, planes, b0 );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
        };
        particles->update_particles( exec_space{}, init_functor );
        particles->update_boundary( exec_space{}, bc );

        // FIXME: use createSolver to switch backend at runtime.
        auto cabana_pd = CabanaPD::createSolverFracture<device_type>(
            inputs, particles, force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}
