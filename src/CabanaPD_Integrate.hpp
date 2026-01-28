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

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Timer.hpp>

namespace CabanaPD
{

template <typename ContactType = NoContact>
class VelocityVerlet;

template <>
class VelocityVerlet<NoContact>
{
  protected:
    double _dt, _half_dt;
    Timer _timer;

  public:
    VelocityVerlet( double dt )
        : _dt( dt )
    {
        _half_dt = 0.5 * dt;
    }

    template <class ExecutionSpace, class ParticlesType>
    void initialHalfStep( ExecutionSpace, ParticlesType& p )
    {
        _timer.start();

        auto u = p.sliceDisplacement();
        auto v = p.sliceVelocity();
        auto f = p.sliceForce();
        auto rho = p.sliceDensity();

        auto dt = _dt;
        auto half_dt = _half_dt;
        auto init_func = KOKKOS_LAMBDA( const int i )
        {
            const double half_dt_m = half_dt / rho( i );
            v( i, 0 ) += half_dt_m * f( i, 0 );
            v( i, 1 ) += half_dt_m * f( i, 1 );
            v( i, 2 ) += half_dt_m * f( i, 2 );
            u( i, 0 ) += dt * v( i, 0 );
            u( i, 1 ) += dt * v( i, 1 );
            u( i, 2 ) += dt * v( i, 2 );
        };
        Kokkos::RangePolicy<ExecutionSpace> policy( p.frozenOffset(),
                                                    p.localOffset() );
        Kokkos::parallel_for( "CabanaPD::VelocityVerlet::Initial", policy,
                              init_func );
        Kokkos::fence();
        _timer.stop();
    }

    template <class ExecutionSpace, class ParticlesType>
    void finalHalfStep( ExecutionSpace, ParticlesType& p )
    {
        _timer.start();

        auto v = p.sliceVelocity();
        auto f = p.sliceForce();
        auto rho = p.sliceDensity();

        auto half_dt = _half_dt;
        auto final_func = KOKKOS_LAMBDA( const int i )
        {
            const double half_dt_m = half_dt / rho( i );
            v( i, 0 ) += half_dt_m * f( i, 0 );
            v( i, 1 ) += half_dt_m * f( i, 1 );
            v( i, 2 ) += half_dt_m * f( i, 2 );
        };
        Kokkos::RangePolicy<ExecutionSpace> policy( p.frozenOffset(),
                                                    p.localOffset() );
        Kokkos::parallel_for( "CabanaPD::VelocityVerlet::Final", policy,
                              final_func );
        Kokkos::fence();
        _timer.stop();
    }

    double timeInit() { return 0.0; };
    auto time() { return _timer.time(); };
};

template <>
class VelocityVerlet<Contact> : public VelocityVerlet<NoContact>
{
    using base_type = VelocityVerlet<NoContact>;
    using base_type::base_type;

  public:
    template <class ExecutionSpace, class ParticlesType>
    void initialHalfStep( ExecutionSpace, ParticlesType& p )
    {
        _timer.start();

        auto u = p.sliceDisplacement();
        auto v = p.sliceVelocity();
        auto f = p.sliceForce();
        auto rho = p.sliceDensity();
        auto u_neigh = p.sliceDisplacementNeighborBuild();

        auto dt = _dt;
        auto half_dt = _half_dt;
        auto init_func = KOKKOS_LAMBDA( const int i, double& max_u )
        {
            const double half_dt_m = half_dt / rho( i );
            v( i, 0 ) += half_dt_m * f( i, 0 );
            v( i, 1 ) += half_dt_m * f( i, 1 );
            v( i, 2 ) += half_dt_m * f( i, 2 );
            u( i, 0 ) += dt * v( i, 0 );
            u( i, 1 ) += dt * v( i, 1 );
            u( i, 2 ) += dt * v( i, 2 );

            auto u_mag = Kokkos::hypot( u( i, 0 ) - u_neigh( i, 0 ),
                                        u( i, 1 ) - u_neigh( i, 1 ),
                                        u( i, 2 ) - u_neigh( i, 2 ) );
            if ( u_mag > max_u )
                max_u = u_mag;
        };
        Kokkos::RangePolicy<ExecutionSpace> policy( p.frozenOffset(),
                                                    p.localOffset() );
        double max_displacement;
        Kokkos::parallel_reduce( "CabanaPD::VelocityVerlet::Initial", policy,
                                 init_func,
                                 Kokkos::Max<double>( max_displacement ) );

        Kokkos::fence();
        p.setMaxDisplacement( max_displacement );
        _timer.stop();
    }

  protected:
    using base_type::_dt;
    using base_type::_half_dt;
    using base_type::_timer;
};

// template <typename ExecutionSpace, typename ParticleType, typename
// FictitiousMassType,typename InitialVelocityType, typename ContactType =
// NoContact> class ADRIntegrator;
//  Single Responsibility Principle: Integrate d^2u/dt^2 = Force in time with
//  Adaptive Dynamic Relaxation Open-Closed Principle: Can be extended by
//  wrapping Liskov-Substitution: no inheritance Interface Segregation: uses "I
//  need interfaces" and uses all passed values Dependency-Inversion: We do not
//  depend on how the entities we are integrating are stored, only depend on "I
//  need interface" for iteration via parallel_for. Doing one integrator per
//  entity is suspected to lead to performance penalties

template <typename ExecutionSpace, typename FictitiousMassType,
          typename InitialVelocityType, int SpatialDimension = 3>
struct ADRIntegrator
{
    static constexpr int dim = SpatialDimension;
    using force_storage_base_type = double[dim];
    using velocity_storage_base_type = double[dim];

    Kokkos::View<force_storage_base_type*, ExecutionSpace> _forces_last_step;
    Kokkos::View<velocity_storage_base_type*, ExecutionSpace>
        _velocities_last_step;
    FictitiousMassType _fictitious_mass;
    InitialVelocityType _initial_velocity_type;
    double _delta_t;

  public:
    ADRIntegrator( ExecutionSpace const& exec_space,
                   FictitiousMassType const& fictitious_mass,
                   InitialVelocityType const& initial_velocity,
                   size_t num_masses, double dt )
        : _forces_last_step(
              Kokkos::View<force_storage_base_type*, ExecutionSpace>(
                  Kokkos::view_alloc( exec_space, Kokkos::WithoutInitializing,
                                      "Forces_Last_Step" ),
                  num_masses ) )
        , _velocities_last_step(
              Kokkos::View<velocity_storage_base_type*, ExecutionSpace>(
                  Kokkos::view_alloc( exec_space, Kokkos::WithoutInitializing,
                                      "Velocities_Last_Step" ),
                  num_masses ) )
        , _fictitious_mass( fictitious_mass )
        , _initial_velocity_type( initial_velocity )
        , _delta_t( dt )
    {
    }

    void reset( ExecutionSpace const& exec_space ) const
    {
        Kokkos::parallel_for(
            "ADRIntegrator::reset",
            Kokkos::RangePolicy( 0, _velocities_last_step.extent( 0 ) ),
            KOKKOS_CLASS_LAMBDA( int64_t index ) {
                for ( int i = 0; i < dim; ++i )
                    _velocities_last_step( index, i ) =
                        _initial_velocity_type( index, i );
            } );
        Kokkos::fence( "ADRIntegrator::Fence::reset" );
    }

    template <typename ForceType>
    void initialStep( ExecutionSpace const& exec_space,
                      ForceType const& forces ) const
    {
        Kokkos::parallel_for(
            "ADRIntegrator::initialStep",
            Kokkos::RangePolicy( 0, _forces_last_step.extent( 0 ) ),
            KOKKOS_CLASS_LAMBDA( int64_t index ) {
                for ( int i = 0; i < dim; ++i )
                    _forces_last_step( index, i ) = forces( index, i );
            } );
        Kokkos::fence( "ADRIntegrator::Fence::intialStep" );
    }
    template <typename ForceType, typename VelocityType,
              typename DisplacementType>
    void finalStep( ExecutionSpace, ForceType const& forces,
                    VelocityType const& velocities,
                    DisplacementType const& displacements ) const
    {
        Kokkos::parallel_for(
            "ADRIntegrator::finalStep",
            Kokkos::RangePolicy( 0, _forces_last_step.extent( 0 ) ),
            KOKKOS_CLASS_LAMBDA( int64_t index ) {
                double mass[dim];
                double stiffness[dim];
                double damping_numerator = 0.;
                double damping_denominator = 0.;
                // compute damping coefficient
                for ( int i = 0; i < dim; ++i )
                {
                    mass[i] = _fictitious_mass( index, i );
                    stiffness[i] = ( -forces( index, i ) +
                                     _forces_last_step( index, i ) ) /
                                   mass[i] / _delta_t /
                                   _velocities_last_step( index, i );
                    damping_numerator += displacements( index, i ) *
                                         stiffness[i] *
                                         displacements( index, i );
                    damping_denominator +=
                        displacements( index, i ) * displacements( index, i );
                }
                double c_damping = 2.0 * Kokkos::sqrt( damping_numerator /
                                                       damping_denominator );

                // update velocity with old velocity and damping coefficient
                double velocity_denomiator = 2.0 + c_damping * _delta_t;
                for ( int i = 0; i < dim; ++i )
                {
                    // update velocity
                    velocities( index, i ) =
                        ( ( 2.0 - c_damping * _delta_t ) *
                              _velocities_last_step( index, i ) +
                          2.0 * _delta_t * forces( index, i ) / mass[i] ) /
                        velocity_denomiator;

                    // update displacement with velocity
                    displacements( index, i ) +=
                        _delta_t * velocities( index, i );

                    // store current velocity as old velocity for next step
                    _velocities_last_step( index, i ) = velocities( index, i );
                }
            } );
        Kokkos::fence( "ADRIntegrator::Fence::finalStep" );
    }
};

struct ADRFictitiousMass
{
    double _delta_t;
    double _horizon;
    double _delta_x;
    double _c;
    double _safety_factor;

    template <typename IndexType>
    double KOKKOS_FUNCTION operator()( IndexType index, int dim ) const
    {
        return _safety_factor *
               ( _delta_t * _delta_t * Kokkos::numbers::pi * _horizon *
                 _horizon * _horizon * _c ) /
               ( 3.0 * _delta_x );
    }
};

template <typename ForcesType, typename FictitiousMassType>
struct ADRInitialVelocity
{
    ForcesType _forces;
    FictitiousMassType _fictitious_mass;
    double _delta_t;

    template <typename IndexType>
    auto KOKKOS_FUNCTION operator()( IndexType index, int dim ) const
    {
        return ( _forces( index, dim ) * _delta_t ) / 2.0 /
               _fictitious_mass( index, dim );
    }
};

template <typename ForcesType, typename FictitiousMassType>
ADRInitialVelocity( ForcesType, FictitiousMassType, double )
    -> ADRInitialVelocity<ForcesType, FictitiousMassType>;

} // namespace CabanaPD

#endif
