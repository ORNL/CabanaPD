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

        // Plate dimensions (m)
        double height = 0.1;
        double width = 0.04;
        double thickness = 0.002;

        // Domain
        std::array<int, 3> num_cell = { 300, 121, 6 }; // 300 x 120 x 6
        double low_x = -0.5 * height;
        double low_y = -0.5 * width;
        double low_z = -0.5 * thickness;
        double high_x = 0.5 * height;
        double high_y = 0.5 * width;
        double high_z = 0.5 * thickness;
        std::array<double, 3> low_corner = { low_x, low_y, low_z };
        std::array<double, 3> high_corner = { high_x, high_y, high_z };

        // Time
        double t_final = 43e-6;
        double dt = 6.7e-8;
        double output_frequency = 5;

        // Material constants
        //double E = 72e+9;                      // [Pa]
        double E = 191e+9;                      // [Pa]
	double nu = 0.25;                      // unitless
        double K = E / ( 3 * ( 1 - 2 * nu ) ); // [Pa]
        //double rho0 = 2440;                    // [kg/m^3]
        double rho0 = 8000;                    // [kg/m^3]
	//double G0 = 3.8;                       // [J/m^2]
	double G0 = 42408;                       // [J/m^2]

        // PD horizon
        double delta = 0.001;

        // FIXME: set halo width based on delta
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cell[0] ) );
        int halo_width = m + 1;

        // Prenotch
        //double L_prenotch = height / 2.0;
        double L_prenotch = width / 2.0;
	//double y_prenotch1 = 0.0;
	double x_prenotch1 = 0.0;
        //Kokkos::Array<double, 3> p01 = { low_x, y_prenotch1, low_z };
	Kokkos::Array<double, 3> p01 = { x_prenotch1, high_y, low_z };
        Kokkos::Array<double, 3> v1 = { 0, -L_prenotch, 0 };
        Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
        Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_positions = { p01 };
        CabanaPD::Prenotch<1> prenotch( v1, v2, notch_positions );

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
            inputs.num_cells, halo_width );

        // Define particle initialization.
        auto x = particles->slice_x();
        auto v = particles->slice_v();
        auto f = particles->slice_f();
        auto rho = particles->slice_rho();
        auto nofail = particles->slice_nofail();

        // Relying on uniform grid here.
        //double dy = particles->dy;
        double dx = particles->dx;
	// double b0 = 2e6 / dy; // Pa
        double b0 = 100e6 / dy; // Pa (100 MPa)

        //CabanaPD::RegionBoundary plane1( low_x, high_x, low_y - dy, low_y + dy,
        //                                 low_z, high_z );
        //CabanaPD::RegionBoundary plane2( low_x, high_x, high_y - dy,
        //                                 high_y + dy, low_z, high_z );
        
	CabanaPD::RegionBoundary plane1( low_x-dx, low_x+dx, low_y, high_y,
                                         low_z, high_z );
        CabanaPD::RegionBoundary plane2( high_x-dx, high_x+dx, low_y, high_y,
					 low_z, high_z );

	std::vector<CabanaPD::RegionBoundary> planes = { plane1, plane2 };
        auto bc =
            createBoundaryCondition( CabanaPD::ForceUpdateBCTag{}, exec_space{},
                                     *particles, planes, b0 );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
            // Set the no-fail zone.
           // if ( x( pid, 1 ) <= plane1.low_y + delta + 1e-10 ||
           //      x( pid, 1 ) >= plane2.high_y - delta - 1e-10 )
           //     nofail( pid ) = 1;
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
