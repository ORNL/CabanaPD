/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef COMM_H
#define COMM_H

#include <string>

#include "mpi.h"

#include <Cabana_Grid.hpp>

#include <CabanaPD_Timer.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <std::size_t Size, class Scalar>
auto vectorToArray( std::vector<Scalar> vector )
{
    Kokkos::Array<Scalar, Size> array;
    for ( std::size_t i = 0; i < Size; ++i )
        array[i] = vector[i];
    return array;
}

// Functor to determine which particles should be ghosted with grid.
template <class MemorySpace, class LocalGridType>
struct HaloIds
{
    static constexpr std::size_t num_space_dim = LocalGridType::num_space_dim;
    // FIXME: 2d
    static constexpr int topology_size = 26;

    using memory_space = MemorySpace;

    int _min_halo;

    using coord_type = Kokkos::Array<double, num_space_dim>;
    Kokkos::Array<coord_type, topology_size> _min_coord;
    Kokkos::Array<coord_type, topology_size> _max_coord;

    Kokkos::Array<int, topology_size> _device_topology;

    using DestinationRankView = typename Kokkos::View<int*, memory_space>;
    using CountView =
        typename Kokkos::View<int, Kokkos::LayoutRight, memory_space,
                              Kokkos::MemoryTraits<Kokkos::Atomic>>;
    CountView _send_count;
    DestinationRankView _destinations;
    DestinationRankView _ids;

    template <class PositionSliceType>
    HaloIds( const LocalGridType& local_grid,
             const PositionSliceType& positions, const int minimum_halo_width,
             const int max_export_guess )
    {
        _destinations = DestinationRankView(
            Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
            max_export_guess );
        _ids = DestinationRankView(
            Kokkos::ViewAllocateWithoutInitializing( "ids" ),
            max_export_guess );
        _send_count = CountView( "halo_send_count" );

        // Check within the halo width, within the local domain.
        _min_halo = minimum_halo_width;

        auto topology = Cabana::Grid::getTopology( local_grid );
        _device_topology = vectorToArray<topology_size>( topology );

        // Get the neighboring mesh bounds (only needed once unless load
        // balancing).
        neighborBounds( local_grid );

        build( positions );
    }

    // Find the bounds of each neighbor rank and store for determining which
    // ghost particles to communicate.
    void neighborBounds( const LocalGridType& local_grid )
    {
        const auto& local_mesh =
            Cabana::Grid::createLocalMesh<Kokkos::HostSpace>( local_grid );

        Kokkos::Array<Cabana::Grid::IndexSpace<4>, topology_size> index_spaces;

        // Store all neighboring shared index space mesh bounds so we only have
        // to launch one kernel during the actual ghost search.
        int n = 0;
        for ( int k = -1; k < 2; ++k )
        {
            for ( int j = -1; j < 2; ++j )
            {
                for ( int i = -1; i < 2; ++i, ++n )
                {
                    if ( i != 0 || j != 0 || k != 0 )
                    {
                        int neighbor_rank = local_grid.neighborRank( i, j, k );
                        // Potentially invalid neighbor ranks (non-periodic
                        // global boundary)
                        if ( neighbor_rank != -1 )
                        {
                            auto sis = local_grid.sharedIndexSpace(
                                Cabana::Grid::Own(), Cabana::Grid::Cell(), i, j,
                                k, _min_halo );
                            auto min_ind = sis.min();
                            auto max_ind = sis.max();
                            local_mesh.coordinates( Cabana::Grid::Node(),
                                                    min_ind.data(),
                                                    _min_coord[n].data() );
                            local_mesh.coordinates( Cabana::Grid::Node(),
                                                    max_ind.data(),
                                                    _max_coord[n].data() );
                        }
                    }
                }
            }
        }
    }

    //---------------------------------------------------------------------------//
    // Locate particles within the local grid and determine if any from this
    // rank need to be ghosted to one (or more) of the 26 neighbor ranks,
    // keeping track of destination rank and index.
    template <class PositionSliceType, class UserFunctor>
    void build( const PositionSliceType& positions, UserFunctor user_functor )
    {
        using execution_space = typename PositionSliceType::execution_space;

        // Local copies of member variables for lambda capture.
        auto send_count = _send_count;
        auto destinations = _destinations;
        auto ids = _ids;
        auto device_topology = _device_topology;
        auto min_coord = _min_coord;
        auto max_coord = _max_coord;

        // Look for ghosts within the halo width of the local mesh boundary,
        // potentially for each of the 26 neighbors cells.
        // Do this one neighbor rank at a time so that sends are contiguous.
        auto ghost_search = KOKKOS_LAMBDA( const int p )
        {
            for ( std::size_t n = 0; n < topology_size; n++ )
            {
                // Potentially invalid neighbor ranks (non-periodic global
                // boundary)
                if ( device_topology[n] != -1 )
                {
                    // Check the if particle is both in the owned
                    // space and the ghosted space of this neighbor
                    // (ignore the current cell).
                    bool within_halo = false;
                    if ( positions( p, 0 ) > min_coord[n][0] &&
                         positions( p, 0 ) < max_coord[n][0] &&
                         positions( p, 1 ) > min_coord[n][1] &&
                         positions( p, 1 ) < max_coord[n][1] &&
                         positions( p, 2 ) > min_coord[n][2] &&
                         positions( p, 2 ) < max_coord[n][2] )
                        within_halo = true;
                    if ( within_halo )
                    {
                        double px[3] = { positions( p, 0 ), positions( p, 1 ),
                                         positions( p, 2 ) };
                        // Let the user restrict to a subset of the boundary.
                        bool create_ghost = user_functor( p, px );
                        if ( create_ghost )
                        {
                            const std::size_t sc = send_count()++;
                            // If the size of the arrays is exceeded,
                            // keep counting to resize and fill next.
                            if ( sc < destinations.extent( 0 ) )
                            {
                                // Keep the destination MPI rank.
                                destinations( sc ) = device_topology[n];
                                // Keep the particle ID.
                                ids( sc ) = p;
                            }
                        }
                    }
                }
            }
        };

        auto policy =
            Kokkos::RangePolicy<execution_space>( 0, positions.size() );
        Kokkos::parallel_for( "CabanaPD::Comm::GhostSearch", policy,
                              ghost_search );
        Kokkos::fence();
    }

    template <class PositionSliceType>
    void build( const PositionSliceType& positions )
    {
        auto empty_functor = KOKKOS_LAMBDA( const int, const double[3] )
        {
            return true;
        };
        build( positions, empty_functor );
    }

    template <class PositionSliceType>
    void rebuild( const PositionSliceType& positions )
    {
        // Resize views to actual send sizes.
        int dest_size = _destinations.extent( 0 );
        int dest_count = 0;
        Kokkos::deep_copy( dest_count, _send_count );
        if ( dest_count != dest_size )
        {
            Kokkos::resize( _destinations, dest_count );
            Kokkos::resize( _ids, dest_count );
        }

        // If original view sizes were exceeded, only counting was done so
        // we need to rerun.
        if ( dest_count > dest_size )
        {
            Kokkos::deep_copy( _send_count, 0 );
            build( positions );
        }
    }
};

template <class ParticleType, class ModelType, class ThermalType>
class Comm;

// FIXME: extract model from ParticleType instead.
template <class ParticleType>
class Comm<ParticleType, PMB, TemperatureIndependent>
{
  public:
    int mpi_size = -1;
    int mpi_rank = -1;
    int max_export;

    using memory_space = typename ParticleType::memory_space;
    using halo_type = Cabana::Halo<memory_space>;
    using gather_u_type =
        Cabana::Gather<halo_type, typename ParticleType::aosoa_u_type>;
    std::shared_ptr<gather_u_type> gather_u;
    std::shared_ptr<halo_type> halo;

    Comm( ParticleType& particles, int max_export_guess = 100 )
        : max_export( max_export_guess )
    {
        _init_timer.start();
        auto local_grid = particles.local_grid;
        MPI_Comm_size( local_grid->globalGrid().comm(), &mpi_size );
        MPI_Comm_rank( local_grid->globalGrid().comm(), &mpi_rank );

        auto positions = particles.sliceReferencePosition();
        // Get all 26 neighbor ranks.
        auto halo_width = local_grid->haloCellWidth();
        auto topology = Cabana::Grid::getTopology( *local_grid );

        // Determine which particles need to be ghosted to neighbors.
        // FIXME: set halo width based on cutoff distance.
        auto halo_ids =
            createHaloIds( *local_grid, positions, halo_width, max_export );
        // Rebuild if needed.
        halo_ids.rebuild( positions );

        // Create the Cabana Halo.
        halo = std::make_shared<halo_type>(
            local_grid->globalGrid().comm(), particles.localOffset(),
            halo_ids._ids, halo_ids._destinations, topology );

        particles.resize( halo->numLocal(), halo->numGhost() );

        // Only use this interface because we don't need to recommunicate
        // positions, volumes, or no-fail region.
        Cabana::gather( *halo, particles.getReferencePosition().aosoa() );
        Cabana::gather( *halo, particles._aosoa_vol );

        gather_u = std::make_shared<gather_u_type>( *halo, particles._aosoa_u );
        gather_u->apply();

        _init_timer.stop();
    }

    auto size() { return mpi_size; }
    auto rank() { return mpi_rank; }

    // Determine which particles should be ghosted, reallocating and recounting
    // if needed.
    template <class LocalGridType, class PositionSliceType>
    auto createHaloIds( const LocalGridType& local_grid,
                        const PositionSliceType& positions,
                        const int min_halo_width, const int max_export )
    {
        return HaloIds<typename PositionSliceType::memory_space, LocalGridType>(
            local_grid, positions, min_halo_width, max_export );
    }

    // We assume here that the particle count has not changed and no resize
    // is necessary.
    void gatherDisplacement()
    {
        _timer.start();
        gather_u->apply();
        _timer.stop();
    }
    // No-op to make solvers simpler.
    void gatherDilatation() {}
    void gatherWeightedVolume() {}

    auto timeInit() { return _init_timer.time(); };
    auto time() { return _timer.time(); };

  protected:
    Timer _init_timer;
    Timer _timer;
};

template <class ParticleType>
class Comm<ParticleType, LPS, TemperatureIndependent>
    : public Comm<ParticleType, PMB, TemperatureIndependent>
{
  public:
    using base_type = Comm<ParticleType, PMB, TemperatureIndependent>;
    using memory_space = typename base_type::memory_space;
    using halo_type = typename base_type::halo_type;
    using base_type::gather_u;
    using base_type::halo;

    using base_type::_init_timer;
    using base_type::_timer;

    using gather_m_type =
        Cabana::Gather<halo_type, typename ParticleType::aosoa_m_type>;
    using gather_theta_type =
        Cabana::Gather<halo_type, typename ParticleType::aosoa_theta_type>;
    std::shared_ptr<gather_m_type> gather_m;
    std::shared_ptr<gather_theta_type> gather_theta;

    Comm( ParticleType& particles, int max_export_guess = 100 )
        : base_type( particles, max_export_guess )
    {
        _init_timer.start();

        gather_m = std::make_shared<gather_m_type>( *halo, particles._aosoa_m );
        gather_theta = std::make_shared<gather_theta_type>(
            *halo, particles._aosoa_theta );

        particles.resize( halo->numLocal(), halo->numGhost() );
        _init_timer.stop();
    }
    ~Comm() {}

    void gatherDilatation()
    {
        _timer.start();
        gather_theta->apply();
        _timer.stop();
    }
    void gatherWeightedVolume()
    {
        _timer.start();
        gather_m->apply();
        _timer.stop();
    }
};

template <class ParticleType>
class Comm<ParticleType, PMB, TemperatureDependent>
    : public Comm<ParticleType, PMB, TemperatureIndependent>
{
  public:
    using base_type = Comm<ParticleType, PMB, TemperatureIndependent>;
    using memory_space = typename base_type::memory_space;
    using halo_type = typename base_type::halo_type;
    using base_type::halo;

    using gather_temp_type =
        Cabana::Gather<halo_type, typename ParticleType::aosoa_temp_type>;
    std::shared_ptr<gather_temp_type> gather_temp;

    Comm( ParticleType& particles, int max_export_guess = 100 )
        : base_type( particles, max_export_guess )
    {
        gather_temp =
            std::make_shared<gather_temp_type>( *halo, particles._aosoa_temp );
        particles.resize( halo->numLocal(), halo->numGhost() );
    }

    void gatherTemperature() { gather_temp->apply(); }
};

} // namespace CabanaPD

#endif
