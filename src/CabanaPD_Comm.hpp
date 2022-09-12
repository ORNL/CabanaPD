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

#include <Cajita.hpp>

namespace CabanaPD
{

// Functor to determine which particles should be ghosted with Cajita grid.
template <class LocalGridType, class PositionSliceType>
struct HaloIds
{
    static constexpr std::size_t num_space_dim = LocalGridType::num_space_dim;

    int _min_halo;
    int _neighbor_rank;

    using device_type = typename PositionSliceType::device_type;
    using pos_value = typename PositionSliceType::value_type;

    using DestinationRankView = typename Kokkos::View<int*, device_type>;
    using CountView =
        typename Kokkos::View<int, Kokkos::LayoutRight, device_type,
                              Kokkos::MemoryTraits<Kokkos::Atomic>>;
    CountView _send_count;
    DestinationRankView _destinations;
    DestinationRankView _ids;
    PositionSliceType _positions;

    Kokkos::Array<int, num_space_dim> _ijk;
    Kokkos::Array<double, num_space_dim> _min_coord;
    Kokkos::Array<double, num_space_dim> _max_coord;

    HaloIds( const LocalGridType& local_grid,
             const PositionSliceType& positions, const int minimum_halo_width,
             const int max_export_guess )
    {
        _positions = positions;
        _destinations = DestinationRankView(
            Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
            max_export_guess );
        _ids = DestinationRankView(
            Kokkos::ViewAllocateWithoutInitializing( "ids" ),
            max_export_guess );
        _send_count = CountView( "halo_send_count" );

        // Check within the halo width, within the local domain.
        _min_halo = minimum_halo_width;

        build( local_grid );
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int p ) const
    {
        // Check the if particle is both in the owned space
        // and the ghosted space of this neighbor (ignore
        // the current cell).
        bool within_halo = false;
        if ( _positions( p, 0 ) > _min_coord[0] &&
             _positions( p, 0 ) < _max_coord[0] &&
             _positions( p, 1 ) > _min_coord[1] &&
             _positions( p, 1 ) < _max_coord[1] &&
             _positions( p, 2 ) > _min_coord[2] &&
             _positions( p, 2 ) < _max_coord[2] )
            within_halo = true;
        if ( within_halo )
        {
            const std::size_t sc = _send_count()++;
            // If the size of the arrays is exceeded, keep
            // counting to resize and fill next.
            if ( sc < _destinations.extent( 0 ) )
            {
                // Keep the destination MPI rank.
                _destinations( sc ) = _neighbor_rank;
                // Keep the particle ID.
                _ids( sc ) = p;
            }
        }
    }

    //---------------------------------------------------------------------------//
    // Locate particles within the local grid and determine if any from this
    // rank need to be ghosted to one (or more) of the 26 neighbor ranks,
    // keeping track of destination rank and index in the container.
    void build( const LocalGridType& local_grid )
    {
        using execution_space = typename PositionSliceType::execution_space;
        const auto& local_mesh =
            Cajita::createLocalMesh<Kokkos::HostSpace>( local_grid );

        auto policy =
            Kokkos::RangePolicy<execution_space>( 0, _positions.size() );

        // Add a ghost if this particle is near the local boundary, potentially
        // for each of the 26 neighbors cells. Do this one neighbor rank at a
        // time so that sends are contiguous.
        auto topology = Cajita::Impl::getTopology( local_grid );
        auto unique_topology = Cabana::Impl::getUniqueTopology( topology );
        for ( std::size_t ar = 0; ar < unique_topology.size(); ar++ )
        {
            int nr = 0;
            for ( int k = -1; k < 2; ++k )
            {
                for ( int j = -1; j < 2; ++j )
                {
                    for ( int i = -1; i < 2; ++i, ++nr )
                    {
                        if ( i != 0 || j != 0 || k != 0 )
                        {
                            _neighbor_rank = topology[nr];
                            if ( _neighbor_rank == unique_topology[ar] )
                            {
                                auto sis = local_grid.sharedIndexSpace(
                                    Cajita::Own(), Cajita::Cell(), i, j, k,
                                    _min_halo );
                                auto min_ind = sis.min();
                                auto max_ind = sis.max();
                                local_mesh.coordinates( Cajita::Node(),
                                                        min_ind.data(),
                                                        _min_coord.data() );
                                local_mesh.coordinates( Cajita::Node(),
                                                        max_ind.data(),
                                                        _max_coord.data() );
                                _ijk = { i, j, k };
                                Kokkos::parallel_for( "get_halo_ids", policy,
                                                      *this );
                                Kokkos::fence();
                            }
                        }
                    }
                }
            }
        }
    }

    void rebuild( const LocalGridType& local_grid )
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
            build( local_grid );
        }
    }
};

template <class DeviceType>
class Comm
{
  public:
    int mpi_size = -1;
    int mpi_rank = -1;
    int max_export;

    using device_type = DeviceType;
    using halo_type = Cabana::Halo<device_type>;
    std::shared_ptr<halo_type> halo;

    template <class ParticleType>
    Comm( ParticleType& particles, int max_export_guess = 100 )
        : max_export( max_export_guess )
    {
        MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );

        auto positions = particles.slice_x();
        // Get all 26 neighbor ranks.
        // FIXME: remove Impl
        auto local_grid = particles.local_grid;
        auto halo_width = local_grid->haloCellWidth();
        auto topology = Cajita::Impl::getTopology( *local_grid );

        // Determine which particles need to be ghosted to neighbors.
        // FIXME: set halo width based on cutoff distance
        auto halo_ids =
            createHaloIds( *local_grid, positions, halo_width, max_export );
        // Rebuild if needed.
        halo_ids.rebuild( *local_grid );

        // Create the Cabana Halo.
        halo = std::make_shared<Cabana::Halo<device_type>>(
            local_grid->globalGrid().comm(), particles.n_local, halo_ids._ids,
            halo_ids._destinations, topology );

        particles.resize( halo->numLocal(), halo->numGhost() );
        particles.gather( *halo );
    }
    ~Comm() {}

    // Determine which particles should be ghosted, reallocating and recounting
    // if needed.
    template <class LocalGridType, class PositionSliceType>
    auto createHaloIds( const LocalGridType& local_grid,
                        const PositionSliceType& positions,
                        const int min_halo_width, const int max_export )
    {
        return HaloIds<LocalGridType, PositionSliceType>(
            local_grid, positions, min_halo_width, max_export );
    }

    // We assume here that the particle count has not changed and no resize
    // is necessary.
    template <class ParticleType>
    void gather( ParticleType& particles )
    {
        particles.gather( *halo );
    }
    template <class ParticleType>
    void gather_theta( ParticleType& particles )
    {
        particles.gather_theta( *halo );
    }
    template <class ParticleType>
    void gather_m( ParticleType& particles )
    {
        particles.gather_m( *halo );
    }
};

} // namespace CabanaPD

#endif
