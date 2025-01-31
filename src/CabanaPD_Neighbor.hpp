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

#ifndef NEIGHBOR_H
#define NEIGHBOR_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_Particles.hpp>

namespace CabanaPD
{
template <typename MemorySpace, typename FractureType = Fracture>
class Neighbor;

template <typename MemorySpace>
class Neighbor<MemorySpace, NoFracture>
{
  public:
    using list_type =
        Cabana::VerletList<MemorySpace, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    using Tag = Cabana::SerialOpTag;
    Tag tag;

  protected:
    using exec_space = typename MemorySpace::execution_space;
    bool _half_neigh;
    list_type _neigh_list;
    double mesh_max[3];
    double mesh_min[3];

    Timer _timer;

  public:
    // Primary constructor: use positions and construct neighbors.
    template <typename ParticleType, typename ModelType>
    Neighbor( const bool half_neigh, const ModelType& model,
              const ParticleType& particles, const double tol = 1e-14 )
        : _half_neigh( half_neigh )
    {
        _timer.start();
        for ( int d = 0; d < particles.dim; d++ )
        {
            mesh_min[d] = particles.ghost_mesh_lo[d];
            mesh_max[d] = particles.ghost_mesh_hi[d];
        }

        if constexpr ( !is_contact<ModelType>::value )
            _neigh_list =
                list_type( particles.sliceReferencePosition(),
                           particles.frozenOffset(), particles.localOffset(),
                           model.cutoff() + tol, 1.0, mesh_min, mesh_max );
        else
            _neigh_list =
                list_type( particles.sliceCurrentPosition(),
                           particles.frozenOffset(), particles.localOffset(),
                           model.cutoff() + tol, 1.0, mesh_min, mesh_max );
        _timer.stop();
    }

    unsigned getMaxLocal()
    {
        auto neigh = _neigh_list;
        unsigned local_max_neighbors;
        auto neigh_max = KOKKOS_LAMBDA( const int, unsigned& max_n )
        {
            max_n = Cabana::NeighborList<list_type>::maxNeighbor( neigh );
        };
        Kokkos::RangePolicy<exec_space> policy( 0, 1 );
        Kokkos::parallel_reduce( policy, neigh_max, local_max_neighbors );
        Kokkos::fence();
        return local_max_neighbors;
    }

    void getStatistics( unsigned& max_neighbors,
                        unsigned long long& total_neighbors )
    {
        auto neigh = _neigh_list;
        unsigned local_max_neighbors;
        unsigned long long local_total_neighbors;
        auto neigh_stats = KOKKOS_LAMBDA( const int, unsigned& max_n,
                                          unsigned long long& total_n )
        {
            max_n = Cabana::NeighborList<list_type>::maxNeighbor( neigh );
            total_n = Cabana::NeighborList<list_type>::totalNeighbor( neigh );
        };
        Kokkos::RangePolicy<exec_space> policy( 0, 1 );
        Kokkos::parallel_reduce( policy, neigh_stats, local_max_neighbors,
                                 local_total_neighbors );
        Kokkos::fence();
        MPI_Reduce( &local_max_neighbors, &max_neighbors, 1, MPI_UNSIGNED,
                    MPI_MAX, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_total_neighbors, &total_neighbors, 1,
                    MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD );
    }

    const auto& list() const { return _neigh_list; }

    // Only rebuild neighbor list as needed.
    template <class ParticleType>
    void update( const ParticleType& particles, const double search_radius,
                 const double radius_extend, const bool require_update = false )
    {
        double max_displacement = particles.getMaxDisplacement();
        if ( max_displacement > radius_extend || require_update )
        {
            _timer.start();
            const auto y = particles.sliceCurrentPosition();
            _neigh_list.build( y, particles.frozenOffset(),
                               particles.localOffset(), search_radius, 1.0,
                               mesh_min, mesh_max );
            // Reset neighbor update displacement.
            const auto u = particles.sliceDisplacement();
            auto u_neigh = particles.sliceDisplacementNeighborBuild();
            Cabana::deep_copy( u_neigh, u );
            _timer.stop();
        }
    }

    template <typename ExecSpace, typename FunctorType, typename ParticleType>
    void iterate( ExecSpace, const FunctorType functor,
                  const ParticleType& particles, const std::string label ) const
    {
        auto policy = makePolicy<ExecSpace>( particles );
        Cabana::neighbor_parallel_for( policy, functor, _neigh_list,
                                       Cabana::FirstNeighborsTag(), tag,
                                       label );
    }

    template <typename ExecSpace, typename FunctorType, typename ParticleType>
    void iterateLinear( ExecSpace, const FunctorType functor,
                        const ParticleType& particles,
                        const std::string label ) const
    {
        auto policy = makePolicy<ExecSpace>( particles );
        Kokkos::parallel_for( label, policy, functor );
    }

    template <typename ExecSpace, typename FunctorType, typename ParticleType,
              typename Scalar>
    void reduce( ExecSpace, const FunctorType functor,
                 const ParticleType& particles, const std::string label,
                 Scalar& total ) const
    {
        auto policy = makePolicy<ExecSpace>( particles );
        Cabana::neighbor_parallel_reduce( policy, functor, _neigh_list,
                                          Cabana::FirstNeighborsTag(), tag,
                                          total, label );
    }

    template <typename ExecSpace, typename FunctorType, typename ParticleType,
              typename... Scalars>
    void reduceLinear( ExecSpace, const FunctorType functor,
                       const ParticleType& particles, const std::string label,
                       Scalars&... scalars ) const
    {
        auto policy = makePolicy<ExecSpace>( particles );
        Kokkos::parallel_reduce( label, policy, functor, scalars... );
    }

    template <typename ExecSpace, typename ParticleType>
    auto makePolicy( const ParticleType& particles ) const
    {
        return Kokkos::RangePolicy<ExecSpace>( particles.frozenOffset(),
                                               particles.localOffset() );
    }

    auto time() { return _timer.time(); };
};

template <typename MemorySpace>
class Neighbor<MemorySpace, Fracture> : public Neighbor<MemorySpace, NoFracture>
{
  public:
    using memory_space = MemorySpace;
    using neighbor_view = typename Kokkos::View<int**, memory_space>;

  protected:
    using base_type = Neighbor<MemorySpace, NoFracture>;
    using base_type::_neigh_list;

    neighbor_view _mu;

  public:
    template <typename ParticleType, typename ModelType>
    Neighbor( const bool half_neigh, const ModelType& model,
              const ParticleType& particles, const double tol = 1e-14 )
        : base_type( half_neigh, model, particles, tol )
    {
        // Create View to track broken bonds.
        // TODO: this could be optimized to ignore frozen particle bonds.
        _mu = neighbor_view(
            Kokkos::ViewAllocateWithoutInitializing( "broken_bonds" ),
            particles.localOffset(), base_type::getMaxLocal() );
        Kokkos::deep_copy( _mu, 1 );
    }

    template <class ExecSpace, class ParticleType, class PrenotchType>
    void prenotch( ExecSpace exec_space, const ParticleType& particles,
                   PrenotchType& prenotch )
    {
        prenotch.create( exec_space, _mu, particles, _neigh_list );
    }

    auto brokenBonds() const { return _mu; }
};

} // namespace CabanaPD

#endif
