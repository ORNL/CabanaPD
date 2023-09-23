#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

template <class ParticleType>
void read_particles( const std::string filename, ParticleType& particles )
{
    // Read particles from csv file.
    std::vector<double> csv_x;
    std::vector<double> csv_y;
    std::vector<double> csv_v;

    std::vector<std::string> row;
    std::string line, word;

    std::fstream file( filename, std::ios::in );
    if ( file.is_open() )
    {
        std::getline( file, line );
        while ( std::getline( file, line ) )
        {
            row.clear();

            std::stringstream str( line );

            while ( std::getline( str, word, ',' ) )
            {
                row.push_back( word );
            }
            csv_x.push_back( std::stod( row[1] ) );
            csv_y.push_back( std::stod( row[2] ) );
            csv_v.push_back( std::stod( row[3] ) );
        }
    }
    else
        throw( "Could not open file." + filename );

    // Create unmanaged Views in order to copy to device.
    Kokkos::View<double*, Kokkos::HostSpace> x_host( csv_x.data() );
    Kokkos::View<double*, Kokkos::HostSpace> y_host( csv_y.data() );
    Kokkos::View<double*, Kokkos::HostSpace> vol_host( csv_v.data() );

    // Copy to the device.
    using memory_space = typename ParticleType::memory_space;
    auto x = Kokkos::create_mirror_view_and_copy( memory_space(), x_host );
    auto y = Kokkos::create_mirror_view_and_copy( memory_space(), y_host );
    auto vol = Kokkos::create_mirror_view_and_copy( memory_space(), vol_host );

    auto px = particles.sliceRefPosition();
    auto pvol = particles.sliceVolume();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto type = particles.sliceType();
    auto rho = particles.sliceDensity();
    auto u = particles.sliceDisplacement();
    auto nofail = particles.sliceNoFail();

    using exec_space = typename memory_space::execution_space;
    Kokkos::parallel_for(
        "copy_to_particles", Kokkos::RangePolicy<exec_space>( 0, x.size() ),
        KOKKOS_LAMBDA( const int pid ) {
            // Set the particle position and volume.
            pvol( pid ) = vol( pid );
            px( pid, 0 ) = x( pid );
            px( pid, 1 ) = y( pid );

            // Initialize everything else to zero.
            px( pid, 2 ) = 0.0;
            for ( int d = 0; d < 3; d++ )
            {
                u( pid, d ) = 0.0;
                v( pid, d ) = 0.0;
                f( pid, d ) = 0.0;
            }
            type( pid ) = 0;
            nofail( pid ) = 0;
            rho( pid ) = 1.0;
        } );
}

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
        double E = 72e+9;                      // [Pa]
        double nu = 0.25;                      // unitless
        double K = E / ( 3 * ( 1 - 2 * nu ) ); // [Pa]
        double rho0 = 2440;                    // [kg/m^3]
        double G0 = 3.8;                       // [J/m^2]

        // PD horizon
        double delta = 0.001;

        // FIXME: set halo width based on delta
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cell[0] ) );
        int halo_width = m + 1;

        // Prenotch
        double L_prenotch = height / 2.0;
        double y_prenotch1 = 0.0;
        Kokkos::Array<double, 3> p01 = { low_x, y_prenotch1, low_z };
        Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
        Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
        Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_positions = { p01 };
        CabanaPD::Prenotch<1> prenotch( v1, v2, notch_positions );

        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
        model_type force_model( delta, K, G0 );
        CabanaPD::Inputs inputs( num_cell, low_corner, high_corner, t_final, dt,
                                 output_frequency );
        inputs.read_args( argc, argv );

        // Default construct to then read particles.
        using device_type = Kokkos::Device<exec_space, memory_space>;
        auto particles = std::make_shared<CabanaPD::Particles<
            device_type, typename model_type::base_model>>();

        // Read particles from file.
        std::string file_name = "file.csv";
        read_particles( file_name, *particles );
        // Update after reading.
        particles->update_after_read( exec_space(), halo_width, num_cell );

        // Define particle initialization.
        auto x = particles->sliceRefPosition();
        auto v = particles->sliceVelocity();
        auto f = particles->sliceForce();
        auto rho = particles->sliceDensity();
        auto nofail = particles->sliceNoFail();

        // Relying on uniform grid here.
        double dy = particles->dy;
        double b0 = 2e6 / dy; // Pa

        CabanaPD::RegionBoundary plane1( low_x, high_x, low_y - dy, low_y + dy,
                                         low_z, high_z );
        CabanaPD::RegionBoundary plane2( low_x, high_x, high_y - dy,
                                         high_y + dy, low_z, high_z );
        std::vector<CabanaPD::RegionBoundary> planes = { plane1, plane2 };
        auto bc =
            createBoundaryCondition( CabanaPD::ForceCrackBranchBCTag{},
                                     exec_space{}, *particles, planes, b0 );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
            // Set the no-fail zone.
            if ( x( pid, 1 ) <= plane1.low_y + delta + 1e-10 ||
                 x( pid, 1 ) >= plane2.high_y - delta - 1e-10 )
                nofail( pid ) = 1;
        };
        particles->updateParticles( exec_space{}, init_functor );

        // FIXME: use createSolver to switch backend at runtime.
        auto cabana_pd = CabanaPD::createSolverFracture<device_type>(
            inputs, particles, force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}
