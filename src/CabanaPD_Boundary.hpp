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
template <class BCIndexSpace, class UserFunctor>
struct BoundaryCondition
{
    BCIndexSpace _index_space;
    UserFunctor _user_functor;
    bool _force_update;

    Timer _timer;

    BoundaryCondition( BCIndexSpace bc_index_space, UserFunctor user,
                       const bool force )
        : _index_space( bc_index_space )
        , _user_functor( user )
        , _force_update( force )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, double time )
    {
        checkParticleCount( _index_space.particle_count,
                            particles.referenceOffset(), "BoundaryCondition" );
        _timer.start();

        auto user = _user_functor;
        auto index_space = _index_space._view;
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
    auto timeInit() { return _index_space.time(); };
};

template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceValueBCTag>
{
    double _value;
    BCIndexSpace _index_space;
    const bool _force_update = true;

    Timer _timer;

    BoundaryCondition( ForceValueBCTag, const double value,
                       BCIndexSpace bc_index_space )
        : _value( value )
        , _index_space( bc_index_space )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, double )
    {
        checkParticleCount( _index_space.particle_count,
                            particles.referenceOffset(), "BoundaryCondition" );

        _timer.start();
        auto f = particles.sliceForce();
        auto index_space = _index_space._view;
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
    auto timeInit() { return _index_space.time(); };
};

template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceUpdateBCTag>
{
    double _value;
    BCIndexSpace _index_space;
    const bool _force_update = true;

    Timer _timer;

    BoundaryCondition( ForceUpdateBCTag, const double value,
                       BCIndexSpace bc_index_space )
        : _value( value )
        , _index_space( bc_index_space )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, double )
    {
        checkParticleCount( _index_space.particle_count,
                            particles.referenceOffset(), "BoundaryCondition" );

        _timer.start();
        auto f = particles.sliceForce();
        auto index_space = _index_space._view;
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
    auto timeInit() { return _index_space.time(); };
};

template <class BCTag, class BCIndexSpace>
BoundaryCondition( BCTag, double, BCIndexSpace )
    -> BoundaryCondition<BCIndexSpace, BCTag>;

template <class BCTag, class ExecSpace, class Particles, class... RegionType>
auto createBoundaryCondition( BCTag tag, const double value,
                              ExecSpace exec_space, Particles particles,
                              RegionType... regions )
{
    BoundaryIndexSpace bc_indices( exec_space, particles, regions... );
    return BoundaryCondition( tag, value, bc_indices );
}

template <class UserFunctor, class ExecSpace, class Particles,
          class... RegionType>
auto createBoundaryCondition( UserFunctor user_functor, ExecSpace exec_space,
                              Particles particles, const bool force_update,
                              RegionType... regions )
{
    BoundaryIndexSpace bc_indices( exec_space, particles, regions... );
    return BoundaryCondition( bc_indices, user_functor, force_update );
}

// Custom index space cases
template <class BCTag, class ExecSpace, class BoundaryParticles>
auto createBoundaryCondition( BCTag, const double value,
                              BoundaryParticles particles )
{
    BoundaryIndexSpaceCustom bc_indices( particles );
    return BoundaryCondition( value, bc_indices );
}

template <class UserFunctor, class BoundaryParticles>
auto createBoundaryCondition( UserFunctor user_functor,
                              BoundaryParticles particles,
                              const bool force_update )
{
    BoundaryIndexSpaceCustom bc_indices( particles );
    return BoundaryCondition( bc_indices, user_functor, force_update );
}

} // namespace CabanaPD

#endif
