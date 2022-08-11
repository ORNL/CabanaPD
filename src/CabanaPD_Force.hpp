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

#ifndef FORCE_H
#define FORCE_H

#include <cmath>

#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{

/******************************************************************************
    Influence function
******************************************************************************/

// FIXME: should enable multiple function options.
KOKKOS_INLINE_FUNCTION
double influence_function( double xi )
{
    double omega = 0;
    // omega = 1;
    omega = 1 / xi;

    return omega;
}

/******************************************************************************
  Force models
******************************************************************************/
struct ForceModel
{
    double delta;

    ForceModel(){};
    ForceModel( const double _delta )
        : delta( _delta ){};
};

/* LPS */

struct LPSModel : public ForceModel
{
    using ForceModel::delta;

    double K;
    double G;
    double theta_coeff;
    double s_coeff;

    LPSModel(){};
    LPSModel( const double delta, const double K, const double G )
        : ForceModel( delta )
    {
        set_param( delta, K, G );
    }

    void set_param( const double _delta, const double _K, const double _G )
    {
        delta = _delta;
        K = _K;
        G = _G;

        theta_coeff = 3 * K - 5 * G;
        s_coeff = 15 * G;
    }
};

struct LinearLPSModel : public LPSModel
{
    using LPSModel::LPSModel;

    using LPSModel::delta;
    using LPSModel::G;
    using LPSModel::K;
    using LPSModel::s_coeff;
    using LPSModel::theta_coeff;
};

/* PMB */

struct PMBModel : public ForceModel
{
    using ForceModel::delta;

    double c;
    double K;

    PMBModel(){};
    PMBModel( const double delta, const double K )
        : ForceModel( delta )
    {
        set_param( delta, K );
    }

    void set_param( const double _delta, const double _K )
    {
        delta = _delta;
        K = _K;
        c = 18.0 * K / ( 3.141592653589793 * delta * delta * delta * delta );
    }
};

struct PMBDamageModel : public PMBModel
{
    using elastic_model = PMBModel;
    using elastic_model::c;
    using elastic_model::delta;
    using elastic_model::K;
    double G0;
    double s0;
    double bond_break_coeff;

    PMBDamageModel() {}
    PMBDamageModel( const double delta, const double K, const double G0 )
        : PMBModel( delta, K )
    {
        set_param( delta, K, G0 );
    }

    void set_param( const double _delta, const double _K, const double _G0 )
    {
        elastic_model::set_param( _delta, _K );
        G0 = _G0;
        s0 = sqrt( 5.0 * G0 / 9.0 / K / delta );
        bond_break_coeff = ( 1 + s0 ) * ( 1 + s0 );
    }
};

struct LinearPMBModel : public PMBModel
{
    using PMBModel::PMBModel;

    using PMBModel::c;
    using PMBModel::delta;
    using PMBModel::K;
};

template <class PosType>
KOKKOS_INLINE_FUNCTION void
getDistanceComponents( const PosType& x, const PosType& u, const int i,
                       const int j, double& xi, double& r, double& s,
                       double& rx, double& ry, double& rz )
{
    // Get the reference positions and displacements.
    const double xi_x = x( j, 0 ) - x( i, 0 );
    const double eta_u = u( j, 0 ) - u( i, 0 );
    const double xi_y = x( j, 1 ) - x( i, 1 );
    const double eta_v = u( j, 1 ) - u( i, 1 );
    const double xi_z = x( j, 2 ) - x( i, 2 );
    const double eta_w = u( j, 2 ) - u( i, 2 );
    rx = xi_x + eta_u;
    ry = xi_y + eta_v;
    rz = xi_z + eta_w;
    r = sqrt( rx * rx + ry * ry + rz * rz );
    xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
    s = ( r - xi ) / xi;
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void getDistance( const PosType& x, const PosType& u,
                                         const int i, const int j, double& xi,
                                         double& r, double& s )
{
    double rx, ry, rz;
    getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void getLinearizedDistanceComponents(
    const PosType& x, const PosType& u, const int i, const int j, double& xi,
    double& s, double& xi_x, double& xi_y, double& xi_z )
{
    // Get the reference positions and displacements.
    xi_x = x( j, 0 ) - x( i, 0 );
    const double eta_u = u( j, 0 ) - u( i, 0 );
    xi_y = x( j, 1 ) - x( i, 1 );
    const double eta_v = u( j, 1 ) - u( i, 1 );
    xi_z = x( j, 2 ) - x( i, 2 );
    const double eta_w = u( j, 2 ) - u( i, 2 );
    xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
    s = ( xi_x * eta_u + xi_y * eta_v + xi_z * eta_w ) / ( xi * xi );
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void
getLinearizedDistance( const PosType& x, const PosType& u, const int i,
                       const int j, double& xi, double& s )
{
    double xi_x, xi_y, xi_z;
    getLinearizedDistanceComponents( x, u, i, j, xi, s, xi_x, xi_y, xi_z );
}

/******************************************************************************
  Force computation LPS
******************************************************************************/
template <class ExecutionSpace, class ForceType>
class Force;

template <class ExecutionSpace>
class Force<ExecutionSpace, LPSModel>
{
  protected:
    bool _half_neigh;
    LPSModel _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const LPSModel model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void initialize( ParticleType& particles, const NeighListType& neigh_list,
                     const ParallelType neigh_op_tag )
    {
        compute_weighted_volume( particles, neigh_list, neigh_op_tag );
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void compute_weighted_volume( ParticleType& particles,
                                  const NeighListType& neigh_list,
                                  const ParallelType neigh_op_tag )
    {
        auto n_local = particles.n_local;
        auto x = particles.slice_x();
        auto u = particles.slice_u();
        const auto vol = particles.slice_vol();
        auto m = particles.slice_m();
        Cabana::deep_copy( m, 0.0 );

        auto weighted_volume = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the reference positions and displacements.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            double m_j = influence_function( xi ) * xi * xi * vol( j );
            m( i ) += m_j;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, weighted_volume, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::compute_full" );
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void compute_force_full( ForceType& f, const PosType& x, const PosType& u,
                             const ParticleType& particles,
                             const NeighListType& neigh_list, const int n_local,
                             ParallelType& neigh_op_tag ) const
    {
        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;

        const auto vol = particles.slice_vol();
        auto theta = particles.slice_theta();
        auto m = particles.slice_m();
        Cabana::deep_copy( theta, 0.0 );

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );

        auto dilatation = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the bond distance, displacement, and stretch
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            double theta_i = influence_function( xi ) * s * xi * xi * vol( j );
            theta( i ) += 3 * theta_i / m( i );
        };

        Cabana::neighbor_parallel_for(
            policy, dilatation, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::compute_dilatation" );

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );
            const double coeff =
                ( theta_coeff * ( theta( i ) / m( i ) + theta( j ) / m( j ) ) +
                  s_coeff * s * ( 1 / m( i ) + 1 / m( j ) ) ) *
                influence_function( xi ) * xi * vol( j );
            fx_i = coeff * rx / r;
            fy_i = coeff * ry / r;
            fz_i = coeff * rz / r;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::compute_full" );
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double compute_energy_full( WType& W, const PosType& x, const PosType& u,
                                const ParticleType& particles,
                                const NeighListType& neigh_list,
                                const int n_local,
                                ParallelType& neigh_op_tag ) const
    {
        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;

        const auto vol = particles.slice_vol();
        const auto theta = particles.slice_theta();
        auto m = particles.slice_m();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );

            double w = ( 1 / num_neighbors ) * 0.5 * theta_coeff / 3 *
                           ( theta( i ) * theta( i ) ) +
                       0.5 * ( s_coeff / m( i ) ) * influence_function( xi ) *
                           s * s * xi * xi * vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForceLPS::compute_energy_full" );

        return strain_energy;
    }
};

template <class ExecutionSpace>
class Force<ExecutionSpace, LinearLPSModel>
{
  protected:
    bool _half_neigh;
    LinearLPSModel _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const LinearLPSModel model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void initialize( ParticleType& particles, const NeighListType& neigh_list,
                     const ParallelType neigh_op_tag )
    {
        compute_weighted_volume( particles, neigh_list, neigh_op_tag );
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void compute_weighted_volume( ParticleType& particles,
                                  const NeighListType& neigh_list,
                                  const ParallelType neigh_op_tag )
    {
        auto n_local = particles.n_local;
        auto x = particles.slice_x();
        auto u = particles.slice_u();
        const auto vol = particles.slice_vol();
        auto m = particles.slice_m();
        Cabana::deep_copy( m, 0.0 );

        auto weighted_volume = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the reference positions and displacements.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            double m_j = influence_function( xi ) * xi * xi * vol( j );
            m( i ) += m_j;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, weighted_volume, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::compute_full" );
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void compute_force_full( ForceType& f, const PosType& x, const PosType& u,
                             const ParticleType& particles,
                             const NeighListType& neigh_list, const int n_local,
                             ParallelType& neigh_op_tag ) const
    {
        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;

        const auto vol = particles.slice_vol();
        auto linear_theta = particles.slice_theta();
        auto m = particles.slice_m();
        Cabana::deep_copy( linear_theta, 0.0 );

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );

        auto dilatation = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the bond distance and stretch
            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            double linear_theta_i =
                influence_function( xi ) * linear_s * xi * xi * vol( j );

            linear_theta( i ) += 3 * linear_theta_i / m( i );
        };

        Cabana::neighbor_parallel_for(
            policy, dilatation, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::compute_dilatation" );

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            // Get the bond distance and stretch
            double xi, linear_s;
            double xi_x, xi_y, xi_z;
            getLinearizedDistanceComponents( x, u, i, j, xi, linear_s, xi_x,
                                             xi_y, xi_z );

            const double coeff =
                ( theta_coeff * ( linear_theta( i ) / m( i ) +
                                  linear_theta( j ) / m( j ) ) +
                  s_coeff * linear_s * ( 1 / m( i ) + 1 / m( j ) ) ) *
                influence_function( xi ) * xi * vol( j );
            fx_i = coeff * xi_x / xi;
            fy_i = coeff * xi_y / xi;
            fz_i = coeff * xi_z / xi;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::compute_full" );
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double compute_energy_full( WType& W, const PosType& x, const PosType& u,
                                const ParticleType& particles,
                                const NeighListType& neigh_list,
                                const int n_local,
                                ParallelType& neigh_op_tag ) const
    {
        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;

        const auto vol = particles.slice_vol();
        const auto linear_theta = particles.slice_theta();
        auto m = particles.slice_m();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {

            // Do we need to recompute linear_theta_i?

            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );

            double w = ( 1 / num_neighbors ) * 0.5 * theta_coeff / 3 *
                           ( linear_theta( i ) * linear_theta( i ) ) +
                       0.5 * ( s_coeff / m( i ) ) * influence_function( xi ) *
                           linear_s * linear_s * xi * xi * vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForceLPS::compute_energy_full" );

        return strain_energy;
    }
};

/******************************************************************************
  Force computation PMB
******************************************************************************/
template <class ExecutionSpace>
class Force<ExecutionSpace, PMBModel>
{
  protected:
    bool _half_neigh;
    PMBModel _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const PMBModel model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void initialize( ParticleType&, NeighListType&, ParallelType )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void compute_force_full( ForceType& f, const PosType& x, const PosType& u,
                             const ParticleType& particles,
                             const NeighListType& neigh_list, const int n_local,
                             ParallelType& neigh_op_tag ) const
    {
        auto c = _model.c;
        const auto vol = particles.slice_vol();

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );
            const double coeff = c * s * vol( j );
            fx_i = coeff * rx / r;
            fy_i = coeff * ry / r;
            fz_i = coeff * rz / r;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForcePMB::compute_full" );
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double compute_energy_full( WType& W, const PosType& x, const PosType& u,
                                const ParticleType& particles,
                                const NeighListType& neigh_list,
                                const int n_local,
                                ParallelType& neigh_op_tag ) const
    {
        auto c = _model.c;
        const auto vol = particles.slice_vol();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Get the bond distance, displacement, and stretch
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            // 1/2 from outside the integral; 1/2 from the integrand (pairwise
            // potential).
            double w = 0.25 * c * s * s * xi * vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForcePMB::compute_energy_full" );

        return strain_energy;
    }
};

template <class ExecutionSpace>
class Force<ExecutionSpace, PMBDamageModel>
{
  protected:
    bool _half_neigh;
    PMBDamageModel _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const PMBDamageModel model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void initialize( ParticleType&, NeighListType&, ParallelType )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    void compute_force_full( ForceType& f, const PosType& x, const PosType& u,
                             const ParticleType& particles,
                             const NeighListType& neigh_list, MuView& mu,
                             const int n_local, ParallelType& ) const
    {
        auto c = _model.c;
        auto break_coeff = _model.bond_break_coeff;
        const auto vol = particles.slice_vol();

        auto force_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                if ( mu( i, n ) > 0 )
                {
                    double fx_i = 0.0;
                    double fy_i = 0.0;
                    double fz_i = 0.0;

                    // Get the reference positions and displacements.
                    std::size_t j =
                        Cabana::NeighborList<NeighListType>::getNeighbor(
                            neigh_list, i, n );
                    const double xi_x = x( j, 0 ) - x( i, 0 );
                    const double eta_u = u( j, 0 ) - u( i, 0 );
                    const double xi_y = x( j, 1 ) - x( i, 1 );
                    const double eta_v = u( j, 1 ) - u( i, 1 );
                    const double xi_z = x( j, 2 ) - x( i, 2 );
                    const double eta_w = u( j, 2 ) - u( i, 2 );
                    const double rx = xi_x + eta_u;
                    const double ry = xi_y + eta_v;
                    const double rz = xi_z + eta_w;
                    const double r2 = rx * rx + ry * ry + rz * rz;
                    const double xi2 = xi_x * xi_x + xi_y * xi_y + xi_z * xi_z;

                    if ( r2 >= break_coeff * xi2 )
                        mu( i, n ) = 0;
                    if ( mu( i, n ) > 0 )
                    {
                        const double r = sqrt( r2 );
                        const double xi = sqrt( xi2 );
                        const double s = ( r - xi ) / xi;
                        const double coeff = c * s * vol( j );
                        double muij = mu( i, n );
                        fx_i = muij * coeff * rx / r;
                        fy_i = muij * coeff * ry / r;
                        fz_i = muij * coeff * rz / r;

                        f( i, 0 ) += fx_i;
                        f( i, 1 ) += fy_i;
                        f( i, 2 ) += fz_i;
                    }
                }
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::ForcePMBDamage::compute_full", policy,
                              force_full );
    }

    template <class PosType, class WType, class DamageType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    double compute_energy_full( WType& W, const PosType& x, const PosType& u,
                                DamageType& phi, const ParticleType& particles,
                                const NeighListType& neigh_list, MuView& mu,
                                const int n_local, ParallelType& ) const
    {
        auto c = _model.c;
        const auto vol = particles.slice_vol();

        auto energy_full = KOKKOS_LAMBDA( const int i, double& Phi )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            double phi_i = 0.0;
            double vol_H_i = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );
                // Get the bond distance, displacement, and stretch
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                // 1/2 from outside the integral; 1/2 from the integrand
                // (pairwise potential).
                double w = mu( i, n ) * 0.25 * c * s * s * xi * vol( j );
                W( i ) += w;

                phi_i += mu( i, n ) * vol( j );
                vol_H_i += vol( j );
            }
            Phi += W( i ) * vol( i );
            phi( i ) = 1 - phi_i / vol_H_i;
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_reduce(
            "CabanaPD::ForcePMBDamage::compute_energy_full", policy,
            energy_full, strain_energy );

        return strain_energy;
    }
};

template <class ExecutionSpace>
class Force<ExecutionSpace, LinearPMBModel>
{
  protected:
    bool _half_neigh;
    LinearPMBModel _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const LinearPMBModel model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void initialize( ParticleType&, NeighListType&, ParallelType )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void compute_force_full( ForceType& f, const PosType& x, const PosType& u,
                             ParticleType& particles,
                             const NeighListType& neigh_list, const int n_local,
                             ParallelType& neigh_op_tag ) const
    {
        auto c = _model.c;
        const auto vol = particles.slice_vol();

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            // Get the bond distance, displacement, and stretch
            double xi, linear_s;
            double xi_x, xi_y, xi_z;
            getLinearizedDistanceComponents( x, u, i, j, xi, linear_s, xi_x,
                                             xi_y, xi_z );

            const double coeff = c * linear_s * vol( j );
            fx_i = coeff * xi_x / xi;
            fy_i = coeff * xi_y / xi;
            fz_i = coeff * xi_z / xi;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLinearPMB::compute_full" );
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double compute_energy_full( WType& W, const PosType& x, const PosType& u,
                                ParticleType& particles,
                                const NeighListType& neigh_list,
                                const int n_local,
                                ParallelType& neigh_op_tag ) const
    {
        auto c = _model.c;
        const auto vol = particles.slice_vol();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Get the bond distance, displacement, and stretch
            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            // 1/2 from outside the integral; 1/2 from the integrand (pairwise
            // potential).
            double w = 0.25 * c * linear_s * linear_s * xi * vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForceLinearPMB::compute_energy_full" );

        return strain_energy;
    }
};

template <class ForceType, class ParticleType, class NeighListType,
          class ParallelType>
void compute_force( const ForceType& force, ParticleType& particles,
                    const NeighListType& neigh_list,
                    const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto x = particles.slice_x();
    auto u = particles.slice_u();
    auto f = particles.slice_f();
    auto f_a = particles.slice_f_a();

    // Reset force.
    Cabana::deep_copy( f, 0.0 );

    // if ( half_neigh )
    // Forces must be atomic for half list
    // compute_force_half( f_a, x, u, neigh_list, n_local,
    //                    neigh_op_tag );

    // Forces only atomic if using team threading
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        force.compute_force_full( f_a, x, u, particles, neigh_list, n_local,
                                  neigh_op_tag );
    else
        force.compute_force_full( f, x, u, particles, neigh_list, n_local,
                                  neigh_op_tag );
    Kokkos::fence();
}

template <class ForceType, class ParticleType, class NeighListType,
          class ParallelType>
double compute_energy( const ForceType force, ParticleType& particles,
                       const NeighListType& neigh_list,
                       const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto x = particles.slice_x();
    auto u = particles.slice_u();
    auto f = particles.slice_f();
    auto W = particles.slice_W();
    auto vol = particles.slice_vol();

    // Reset energy.
    Cabana::deep_copy( W, 0.0 );

    double energy;
    // if ( _half_neigh )
    //    energy = compute_energy_half( force, x, u, neigh_list,
    //                                  n_local, neigh_op_tag );
    // else
    energy = force.compute_energy_full( W, x, u, particles, neigh_list, n_local,
                                        neigh_op_tag );
    Kokkos::fence();

    return energy;
}

// Forces with bond breaking.
template <class ForceType, class ParticleType, class NeighListType,
          class NeighborView, class ParallelType>
void compute_force( const ForceType& force, ParticleType& particles,
                    const NeighListType& neigh_list, NeighborView& mu,
                    const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto x = particles.slice_x();
    auto u = particles.slice_u();
    auto f = particles.slice_f();
    auto f_a = particles.slice_f_a();

    // Reset force.
    Cabana::deep_copy( f, 0.0 );

    // if ( half_neigh )
    // Forces must be atomic for half list
    // compute_force_half( f_a, x, u, neigh_list, n_local,
    //                    neigh_op_tag );

    // Forces only atomic if using team threading
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        force.compute_force_full( f_a, x, u, particles, neigh_list, mu, n_local,
                                  neigh_op_tag );
    else
        force.compute_force_full( f, x, u, particles, neigh_list, mu, n_local,
                                  neigh_op_tag );
    Kokkos::fence();
}

// Energy and damage.
template <class ForceType, class ParticleType, class NeighListType,
          class NeighborView, class ParallelType>
double compute_energy( const ForceType force, ParticleType& particles,
                       const NeighListType& neigh_list, NeighborView& mu,
                       const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto x = particles.slice_x();
    auto u = particles.slice_u();
    auto f = particles.slice_f();
    auto W = particles.slice_W();
    auto phi = particles.slice_phi();

    // Reset energy.
    Cabana::deep_copy( W, 0.0 );

    double energy;
    // if ( _half_neigh )
    //    energy = compute_energy_half( force, x, u, neigh_list,
    //                                  n_local, neigh_op_tag );
    // else
    energy = force.compute_energy_full( W, x, u, phi, particles, neigh_list, mu,
                                        n_local, neigh_op_tag );
    Kokkos::fence();

    return energy;
}

} // namespace CabanaPD

#endif
