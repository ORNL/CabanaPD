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

    template <class ExecSpace, class Particles, class BoundaryType>
    void update( ExecSpace exec_space, Particles particles, BoundaryType plane )
    {
        _indices.update( exec_space, particles, plane );
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType&, double time )
    {
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

    BoundaryCondition( const double value, steering_vector_type bc_indices )
        : _value( value )
        , _indices( bc_indices )
    {
    }

    template <class ExecSpace, class Particles, class BoundaryType>
    void update( ExecSpace exec_space, Particles particles, BoundaryType plane )
    {
        _indices.update( exec_space, particles, plane );
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, double )
    {
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

    BoundaryCondition( const double value, steering_vector_type bc_indices )
        : _value( value )
        , _indices( bc_indices )
    {
    }

    template <class ExecSpace, class Particles, class BoundaryType>
    void update( ExecSpace exec_space, Particles particles, BoundaryType plane )
    {
        _indices.update( exec_space, particles, plane );
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, double )
    {
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

// FIXME: relatively large initial guess for allocation.
template <class BoundaryType, class BCTag, class ExecSpace, class Particles>
auto createBoundaryCondition( BCTag, const double value, ExecSpace exec_space,
                              Particles particles,
                              std::vector<BoundaryType> planes,
                              const double initial_guess = 0.5 )
{
    using memory_space = typename Particles::memory_space;
    auto bc_indices = createParticleSteeringVector( exec_space, particles,
                                                    planes, initial_guess );
    return BoundaryCondition<memory_space, BCTag>( value, bc_indices );
}

// FIXME: relatively large initial guess for allocation.
template <class UserFunctor, class BoundaryType, class ExecSpace,
          class Particles>
auto createBoundaryCondition( UserFunctor user_functor, ExecSpace exec_space,
                              Particles particles,
                              std::vector<BoundaryType> planes,
                              const bool force_update,
                              const double initial_guess = 0.5 )
{
    using memory_space = typename Particles::memory_space;
    auto bc_indices = createParticleSteeringVector( exec_space, particles,
                                                    planes, initial_guess );
    return BoundaryCondition<memory_space, UserFunctor>(
        bc_indices, user_functor, force_update );
}

// Custom index space cases
template <class BCTag, class ExecSpace, class BoundaryParticles>
auto createBoundaryCondition( BCTag, const double value,
                              BoundaryParticles particles )
{
    using memory_space = typename BoundaryParticles::memory_space;
    auto bc_indices = createParticleSteeringVector( particles );
    return BoundaryCondition<memory_space, BCTag>( value, bc_indices );
}

template <class UserFunctor, class BoundaryParticles>
auto createBoundaryCondition( UserFunctor user_functor,
                              BoundaryParticles particles,
                              const bool force_update )
{
    using memory_space = typename BoundaryParticles::memory_space;
    auto bc_indices = createParticleSteeringVector( particles );
    return BoundaryCondition<memory_space, UserFunctor>(
        bc_indices, user_functor, force_update );
}

// Wrappers for single plane cases.
template <class BoundaryType, class BCTag, class ExecSpace, class Particles>
auto createBoundaryCondition( BCTag tag, const double value,
                              ExecSpace exec_space, Particles particles,
                              BoundaryType plane,
                              const double initial_guess = 0.5 )
{
    std::vector<BoundaryType> plane_vec = { plane };
    return createBoundaryCondition( tag, value, exec_space, particles,
                                    plane_vec, initial_guess );
}

template <class UserFunctor, class BoundaryType, class ExecSpace,
          class Particles>
auto createBoundaryCondition( UserFunctor user_functor, ExecSpace exec_space,
                              Particles particles, BoundaryType plane,
                              const bool force_update,
                              const double initial_guess = 0.5 )
{
    std::vector<BoundaryType> plane_vec = { plane };
    return createBoundaryCondition( user_functor, exec_space, particles,
                                    plane_vec, force_update, initial_guess );
}

} // namespace CabanaPD

#endif
