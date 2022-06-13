#ifndef BOUNDARYCONDITION_H
#define BOUNDARYCONDITION_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

namespace CabanaPD
{

struct ZeroBoundary
{
};

struct RegionBoundary
{
    double low_x;
    double high_x;
    double low_y;
    double high_y;
    double low_z;
    double high_z;

    RegionBoundary( const double _low_x, const double _high_x,
                    const double _low_y, const double _high_y,
                    const double _low_z, const double _high_z )
        : low_x( _low_x )
        , high_x( _high_x )
        , low_y( _low_y )
        , high_y( _high_y )
        , low_z( _low_z )
        , high_z( _high_z ){};
};

template <class MemorySpace, class BoundaryType>
struct BoundaryIndexSpace;

template <class MemorySpace>
struct BoundaryIndexSpace<MemorySpace, RegionBoundary>
{
    using index_view_type = Kokkos::View<int*, MemorySpace>;
    index_view_type _index_space;

    template <class ExecSpace, class Particles>
    BoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                        RegionBoundary plane )
    {
        create( exec_space, particles, plane );
    }

    template <class ExecSpace, class Particles>
    void create( ExecSpace, Particles particles, RegionBoundary plane )
    {
        // Guess 10% boundary particles.
        auto index_space =
            index_view_type( "boundary_indices", particles.n_local * 0.1 );
        auto x = particles.slice_x();
        Kokkos::RangePolicy<ExecSpace> policy( 0, particles.n_local );
        auto index_functor =
            KOKKOS_LAMBDA( const std::size_t pid, std::size_t& c )
        {
            if ( x( pid, 0 ) >= plane.low_x && x( pid, 0 ) <= plane.high_x &&
                 x( pid, 1 ) >= plane.low_y && x( pid, 1 ) <= plane.high_y &&
                 x( pid, 2 ) >= plane.low_z && x( pid, 2 ) <= plane.high_z )
            {
                // Resize after count if needed.
                if ( c < index_space.size() )
                {
                    index_space( c ) = pid;
                }
                c += 1;
            }
        };

        std::size_t sum = 0;
        Kokkos::parallel_reduce( "CabanaPD::BC::create", policy, index_functor,
                                 Kokkos::Sum<std::size_t>( sum ) );

        if ( sum > index_space.size() )
        {
            Kokkos::realloc( index_space, sum );
            sum = 0;
            Kokkos::parallel_reduce( "CabanaPD::BC::create", policy,
                                     index_functor,
                                     Kokkos::Sum<std::size_t>( sum ) );
        }

        Kokkos::resize( index_space, sum );
        _index_space = index_space;
    }
};

template <class ExecSpace, class Particles, class BoundaryType>
auto createBoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                               BoundaryType plane )
{
    using memory_space = typename Particles::memory_space;
    return BoundaryIndexSpace<memory_space, BoundaryType>( exec_space,
                                                           particles, plane );
}

class ForceBCTag
{
};

template <class BCIndexSpace, class BCTag>
struct BoundaryCondition;

template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceBCTag>
{
    using view_type = typename BCIndexSpace::index_view_type;
    view_type _index_space;

    BoundaryCondition( BCIndexSpace bc_index_space )
        : _index_space( bc_index_space._index_space )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles )
    {
        auto f = particles.slice_f();
        auto index_space = _index_space;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                auto pid = index_space( b );
                for ( int d = 0; d < 3; d++ )
                    f( pid, d ) = 0.0;
            } );
    }
};

template <class ExecSpace, class Particles, class BoundaryType, class BCTag>
auto createBoundaryCondition( ExecSpace exec_space, Particles particles,
                              BoundaryType plane, BCTag )
{
    using memory_space = typename Particles::memory_space;
    using bc_index_type = BoundaryIndexSpace<memory_space, BoundaryType>;
    bc_index_type bc_indices =
        createBoundaryIndexSpace( exec_space, particles, plane );
    return BoundaryCondition<bc_index_type, BCTag>( bc_indices );
}

} // namespace CabanaPD

#endif
