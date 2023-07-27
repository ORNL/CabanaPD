#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

template <typename MemorySpace, typename ModelType>
auto read_particles( const std::string filename, const double dx,
                     const int halo_width )
{
    // Read particles from csv file.
    // This assumes 2 positions and a volume and ignores the rest.
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
    auto x = Kokkos::create_mirror_view_and_copy( MemorySpace(), x_host );
    auto y = Kokkos::create_mirror_view_and_copy( MemorySpace(), y_host );
    auto vol = Kokkos::create_mirror_view_and_copy( MemorySpace(), vol_host );

    // Copy into a single position View.
    Kokkos::View<double* [3], MemorySpace> position( "custom_position",
                                                     csv_v.size() );
    Kokkos::View<double*, MemorySpace> volume( "custom_volume", csv_v.size() );

    // Using atomic memory for the count.
    using CountView =
        typename Kokkos::View<int, Kokkos::LayoutRight, MemorySpace,
                              Kokkos::MemoryTraits<Kokkos::Atomic>>;
    CountView count( "count" );

    using exec_space = typename MemorySpace::execution_space;
    Kokkos::parallel_for(
        "copy_to_position", Kokkos::RangePolicy<exec_space>( 0, x.size() ),
        KOKKOS_LAMBDA( const int pid ) {
            // Guard against duplicate points. This threaded loop should be
            // faster than checking during the host read.
            for ( int p = 0; p < pid; p++ )
            {
                auto xdiff = x( p ) - x( pid );
                auto ydiff = y( p ) - y( pid );
                if ( ( Kokkos::abs( xdiff ) < 1e-14 ) &&
                     ( Kokkos::abs( ydiff ) < 1e-14 ) )
                    return;
            }
            const std::size_t c = count()++;

            // Set the particle position.
            position( c, 0 ) = x( pid );
            position( c, 1 ) = y( pid );
            position( c, 2 ) = 0.0;
            vol( c ) = vol( pid );
        } );

    std::array<double, 3> dx_array = { dx, dx, dx };
    return CabanaPD::Particles( MemorySpace{}, ModelType{}, position, vol,
                                dx_array, halo_width, exec_space{} );
}

void fileReadExample( const std::string filename )
{
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    CabanaPD::Inputs inputs( filename );

    // Choose force model type.
    using model_type = CabanaPD::PMB;
    // Read particles from file.
    double delta = inputs["horizon"];
    double dx = 0.5;
    int halo_width = 1;
    auto particles = read_particles<memory_space, model_type>(
        inputs["data_file"], dx, halo_width );

    auto x = particles.sliceReferencePosition();
    auto nofail = particles.sliceNoFail();
    double max_bc = particles.local_mesh_hi[1] - delta;
    double min_bc = particles.local_mesh_lo[1] + delta;
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        if ( x( pid, 1 ) <= min_bc + 1e-10 || x( pid, 1 ) >= max_bc - 1e-10 )
            nofail( pid ) = 1;
    };
    particles.updateParticles( exec_space{}, init_functor );

    double E = inputs["elastic_modulus"];
    double K = E / ( 3 * ( 1 - 2 * 0.25 ) );
    double G0 = inputs["fracture_energy"];
    CabanaPD::ForceModel force_model( model_type{}, delta, K, G0 );

    CabanaPD::Region<CabanaPD::RectangularPrism> planeUpper(
        particles.local_mesh_lo[0], particles.local_mesh_hi[0], max_bc,
        particles.local_mesh_hi[1], particles.local_mesh_lo[2],
        particles.local_mesh_hi[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> planeLower(
        particles.local_mesh_lo[0], particles.local_mesh_hi[0],
        particles.local_mesh_lo[1], min_bc, particles.local_mesh_lo[2],
        particles.local_mesh_hi[2] );

    double t_ramp = 1e-3;
    double t_start = 0.0;
    double t_end = std::numeric_limits<double>::max();
    double bc_value = 2.0e4;
    double center = 0.0;
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto bc_functor = KOKKOS_LAMBDA( const int b, const double t )
    {
        // Initial linear ramp.
        const int dim = 0;
        auto current_top = bc_value;
        auto current_bottom = -bc_value;
        if ( t < t_ramp )
        {
            auto t_factor = ( t - t_start ) / ( t_ramp - t_start );
            current_top = bc_value * t_factor;
            current_bottom = -bc_value * t_factor;
        }
        if ( t > t_start && t < t_end )
        {
            if ( x( b, dim ) > center )
                f( b, dim ) += current_top;
            else
                f( b, dim ) += current_bottom;
        }
    };
    auto bc = CabanaPD::createBoundaryCondition(
        bc_functor, exec_space{}, particles, true, planeUpper, planeLower );

    // FIXME: use createSolver to switch backend at runtime.
    CabanaPD::Solver solver( inputs, particles, force_model );
    solver.init( bc );
    solver.run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    fileReadExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
