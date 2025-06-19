
namespace CabanaPD
{
template <typename ParticleType, typename NewParticleType>
void interpolate( const ParticleType& particles,
                  NewParticleType& new_particles )
{
    using memory_space = typename NewParticleType::memory_space;

    // Interpolate to the background grid.
    auto powder_positions = particles.sliceCurrentPosition();
    auto density = particles.sliceDensity();
    auto num_particles = particles.localOffset();
    auto scalar_p2g = Cabana::Grid::createScalarValueP2G( density, 0.125 );
    auto scalar_layout = Cabana::Grid::createArrayLayout(
        new_particles.local_grid, 1, Cabana::Grid::Node() );
    auto scalar_grid_field = Cabana::Grid::createArray<double, memory_space>(
        "scalar_grid_field", scalar_layout );
    auto scalar_halo = Cabana::Grid::createHalo(
        Cabana::Grid::NodeHaloPattern<3>(), new_particles.halo_width,
        *scalar_grid_field );
    Cabana::Grid::p2g( scalar_p2g, powder_positions, num_particles,
                       Cabana::Grid::Spline<0>(), *scalar_halo,
                       *scalar_grid_field );
    Cabana::Grid::Experimental::BovWriter::writeTimeStep(
        "grid_density", 0, 0.0, *scalar_grid_field );

    // Now interpolate to what would be the consolidation particles.
    auto consolidation_positions = new_particles.sliceCurrentPosition();
    auto consolidation_density = new_particles.sliceDensity();
    auto pd_num_particles = new_particles.localOffset();
    auto scalar_value_g2p =
        Cabana::Grid::createScalarValueG2P( consolidation_density, 1.0 );
    Cabana::Grid::g2p( *scalar_grid_field, *scalar_halo,
                       consolidation_positions, pd_num_particles,
                       Cabana::Grid::Spline<0>(), scalar_value_g2p );
}
} // namespace CabanaPD
