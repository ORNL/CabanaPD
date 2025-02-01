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

#ifndef BOUNDARYCONDITION_H
#define BOUNDARYCONDITION_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>

#include <CabanaPD_Geometry.hpp>
#include <CabanaPD_Timer.hpp>

namespace CabanaPD
{

struct ForceValueBCTag
{
};
struct ForceUpdateBCTag
{
};

// Custom boundary condition.
template <class MemorySpace, class UserFunctor>
struct BoundaryCondition
{
    using steering_vector_type = ParticleSteeringVector<MemorySpace>;
    steering_vector_type _indices;
    UserFunctor _user_functor;
    bool _force_update;

    Timer _timer;

    BoundaryCondition( steering_vector_type bc_indices, UserFunctor user,
                       const bool force )
        : _indices( bc_indices )
        , _user_functor( user )
        , _force_update( force )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, double time )
    {
        checkParticleCount( _indices.particle_count,
                            particles.referenceOffset(), "BoundaryCondition" );
        _timer.start();

        auto user = _user_functor;
        auto index_space = _indices._view;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                auto pid = index_space( b );
                user( pid, time );
            } );
        _timer.stop();
    }

    auto forceUpdate() { return _force_update; }

    auto time() { return _timer.time(); };
    auto timeInit() { return _indices.time(); };
};

template <class MemorySpace>
struct BoundaryCondition<MemorySpace, ForceValueBCTag>
{
    double _value;
    using steering_vector_type = ParticleSteeringVector<MemorySpace>;
    steering_vector_type _indices;
    const bool _force_update = true;

    Timer _timer;

    BoundaryCondition( ForceValueBCTag, const double value,
                       steering_vector_type bc_indices )
        : _value( value )
        , _indices( bc_indices )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, double )
    {
        checkParticleCount( _indices.particle_count,
                            particles.referenceOffset(), "BoundaryCondition" );

        _timer.start();
        auto f = particles.sliceForce();
        auto index_space = _indices._view;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        auto value = _value;
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                auto pid = index_space( b );
                for ( int d = 0; d < 3; d++ )
                    f( pid, d ) = value;
            } );
        _timer.stop();
    }

    auto forceUpdate() { return _force_update; }

    auto time() { return _timer.time(); };
    auto timeInit() { return _indices.time(); };
};

template <class MemorySpace>
struct BoundaryCondition<MemorySpace, ForceUpdateBCTag>
{
    double _value;
    using steering_vector_type = ParticleSteeringVector<MemorySpace>;
    steering_vector_type _indices;
    const bool _force_update = true;

    Timer _timer;

    BoundaryCondition( ForceUpdateBCTag, const double value,
                       steering_vector_type bc_indices )
        : _value( value )
        , _indices( bc_indices )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, double )
    {
        checkParticleCount( _indices.particle_count,
                            particles.referenceOffset(), "BoundaryCondition" );

        _timer.start();
        auto f = particles.sliceForce();
        auto index_space = _indices._view;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        auto value = _value;
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                auto pid = index_space( b );
                for ( int d = 0; d < 3; d++ )
                    f( pid, d ) += value;
            } );
        _timer.stop();
    }

    auto forceUpdate() { return _force_update; }

    auto time() { return _timer.time(); };
    auto timeInit() { return _indices.time(); };
};

template <class BCTag, class SteeringVectorType>
BoundaryCondition( BCTag, double, SteeringVectorType )
    -> BoundaryCondition<typename SteeringVectorType::memory_space, BCTag>;

template <class BCTag, class ExecSpace, class Particles, class... RegionType>
auto createBoundaryCondition( BCTag tag, const double value,
                              ExecSpace exec_space, Particles particles,
                              RegionType... regions )
{
    ParticleSteeringVector bc_indices( exec_space, particles, regions... );
    return BoundaryCondition( tag, value, bc_indices );
}

template <class UserFunctor, class ExecSpace, class Particles,
          class... RegionType>
auto createBoundaryCondition( UserFunctor user_functor, ExecSpace exec_space,
                              Particles particles, const bool force_update,
                              RegionType... regions )
{
    ParticleSteeringVector bc_indices( exec_space, particles, regions... );
    return BoundaryCondition( bc_indices, user_functor, force_update );
}

// Custom boundary particle cases.
template <class BCTag, class ExecSpace, class BoundaryParticles>
auto createBoundaryCondition( BCTag, const double value,
                              BoundaryParticles particles )
{
    ParticleSteeringVector bc_indices( particles );
    return BoundaryCondition( value, bc_indices );
}

template <class UserFunctor, class BoundaryParticles>
auto createBoundaryCondition( UserFunctor user_functor,
                              BoundaryParticles particles,
                              const bool force_update )
{
    ParticleSteeringVector bc_indices( particles );
    return BoundaryCondition( bc_indices, user_functor, force_update );
}

} // namespace CabanaPD

#endif
