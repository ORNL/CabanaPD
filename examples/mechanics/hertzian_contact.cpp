#include <CabanaPD.hpp>

#include <Kokkos_Core.hpp>

void hertzianContactExample( const std::string filename )
{
    // ====================================================
    //             Use default Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                Material parameters
    // ====================================================
    double rho0 = inputs["density"];
    double vol = inputs["restitution"];
    double nu = inputs["poisson_ratio"];
    double E = inputs["elastic_modulus"];
    double e = inputs["restitution"];
    double delta = inputs["horizon"];
    delta += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];

    // ====================================================
    //            Custom particle creation
    // ====================================================
    const int num_particles = 2;
    // Purposely using zero-init here.
    Kokkos::View<double* [3], memory_space> position( "custom_position", 2 );
    Kokkos::View<double*, memory_space> volume( "custom_volume", 2 );

    Kokkos::parallel_for(
        "create_particles", Kokkos::RangePolicy<exec_space>( 0, num_particles ),
        KOKKOS_LAMBDA( const int p ) {
            if ( p == 0 )
                position( p, 0 ) = 1.0;
            else
                position( p, 0 ) = -1.0;
            volume( p ) = vol;
        } );

    // ====================================================
    //            Force model
    // ====================================================
    using model_type = CabanaPD::HertzianModel;
    model_type contact_model( delta, nu, E, e );

    // ====================================================
    //                 Particle generation
    // ====================================================
    int halo_width = 1;
    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space{}, position, volume, low_corner, high_corner, num_cells,
        halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles->sliceDensity();
    auto v = particles->sliceVelocity();

    auto init_functor = KOKKOS_LAMBDA( const int p )
    {
        // Density
        rho( p ) = rho0;
        if ( p == 0 )
            v( p, 0 ) = -1.0;
        else
            v( p, 0 ) = 1.0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //  Simulation run
    // ====================================================

    auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
        inputs, particles, contact_model );
    cabana_pd->init();
    cabana_pd->run();
}

// Initialize Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    hertzianContactExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
