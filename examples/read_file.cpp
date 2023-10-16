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
            csv_x.push_back( std::stod( row[0] ) );
            csv_y.push_back( std::stod( row[1] ) );
            csv_v.push_back( std::stod( row[2] ) );
        }
    }
    else
        throw( "Could not open file." + filename );

    // Create unmanaged Views in order to copy to device.
    Kokkos::View<double*, Kokkos::HostSpace> x_host( csv_x.data(),
                                                     csv_x.size() );
    Kokkos::View<double*, Kokkos::HostSpace> y_host( csv_y.data(),
                                                     csv_y.size() );
    Kokkos::View<double*, Kokkos::HostSpace> vol_host( csv_v.data(),
                                                       csv_v.size() );

    // Copy to the device.
    using memory_space = typename ParticleType::memory_space;
    auto x = Kokkos::create_mirror_view_and_copy( memory_space(), x_host );
    auto y = Kokkos::create_mirror_view_and_copy( memory_space(), y_host );
    auto vol = Kokkos::create_mirror_view_and_copy( memory_space(), vol_host );
    // Resize internal variables (no ghosts initially).
    particles.resize( x.size(), 0 );

    auto px = particles.sliceReferencePosition();
    auto pvol = particles.sliceVolume();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto type = particles.sliceType();
    auto rho = particles.sliceDensity();
    auto u = particles.sliceDisplacement();
    auto nofail = particles.sliceNoFail();
    auto damage = particles.sliceDamage();

    using exec_space = typename memory_space::execution_space;
    Kokkos::parallel_for(
        "copy_to_particles", Kokkos::RangePolicy<exec_space>( 0, x.size() ),
        KOKKOS_LAMBDA( const int pid ) {
            // Set the particle position and volume.
            pvol( pid ) = vol( pid );
            px( pid, 0 ) = x( pid );
            px( pid, 1 ) = y( pid );

            // Initialize everything else to zero.
            // Currently psuedo-2d.
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
            damage( pid ) = 0;
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

        // Time
        double t_final = 1e-6;
        double dt = 1e-9;
        double output_frequency = 10;

        // Material constants
        double E = 72e+9;                      // [Pa]
        double nu = 0.25;                      // unitless
        double K = E / ( 3 * ( 1 - 2 * nu ) ); // [Pa]
        double G0 = 3.8;                       // [J/m^2]

        // PD horizon
        double dx = 0.1299999999999999;
        double delta = 3 * dx;
        int halo_width = 1;

        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
        model_type force_model( delta, K, G0 );
        CabanaPD::Inputs<3> inputs( t_final, dt, output_frequency, true );
        inputs.read_args( argc, argv );

        // Default construct to then read particles.
        using device_type = Kokkos::Device<exec_space, memory_space>;
        auto particles = std::make_shared<CabanaPD::Particles<
            device_type, typename model_type::base_model, 3>>();

        // Read particles from file.
        read_particles( inputs.input_file, *particles );
        // Update after reading. Currently requires fixed cell spacing.
        particles->updateAfterRead( exec_space(), halo_width, dx );

        CabanaPD::Prenotch<0> prenotch;

        CabanaPD::RegionBoundary planeUpper( -0.5, 0.5, 4.5, 5, -1, 1 );
        CabanaPD::RegionBoundary planeLower( -0.5, 0.5, -5, -4.5, -1, 1 );

        std::vector<CabanaPD::RegionBoundary> planes = { planeUpper,
                                                         planeLower };
        int bc_dim = 2;
        double center = particles->local_mesh_ext[bc_dim] / 2.0 +
                        particles->local_mesh_lo[bc_dim];
        auto bc = createBoundaryCondition( CabanaPD::ForceSymmetric1dBCTag{},
                                           exec_space{}, *particles, planes,
                                           2e6 / dx / dx, bc_dim, center );

        // FIXME: use createSolver to switch backend at runtime.
        auto cabana_pd = CabanaPD::createSolverFracture<device_type>(
            inputs, particles, force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}
