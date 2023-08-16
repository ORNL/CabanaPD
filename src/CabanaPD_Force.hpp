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
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
/******************************************************************************
  Force models
******************************************************************************/
struct BaseForceModel
{
    double delta;

    BaseForceModel(){};
    BaseForceModel( const double _delta )
        : delta( _delta ){};

    BaseForceModel( const BaseForceModel& model ) { delta = model.delta; }
};

template <typename ModelType, typename DamageType>
struct ForceModel;

/* LPS */
template <>
struct ForceModel<LPS, Elastic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using base_model = LPS;
    using fracture_type = Elastic;

    using BaseForceModel::delta;

    int influence_type;

    double K;
    double G;
    double theta_coeff;
    double s_coeff;

    ForceModel(){};
    ForceModel( const double _delta, const double _K, const double _G,
                const int _influence = 0 )
        : base_type( _delta )
        , influence_type( _influence )
    {
        set_param( _delta, _K, _G );
    }

    void set_param( const double _delta, const double _K, const double _G )
    {
        delta = _delta;
        K = _K;
        G = _G;

        theta_coeff = 3.0 * K - 5.0 * G;
        s_coeff = 15.0 * G;
    }

    KOKKOS_INLINE_FUNCTION double influence_function( double xi ) const
    {
        if ( influence_type == 1 )
            return 1.0 / xi;
        else
            return 1.0;
    }
};

template <>
struct ForceModel<LPS, Fracture> : public ForceModel<LPS, Elastic>
{
    using base_type = ForceModel<LPS, Elastic>;
    using base_model = LPS;
    using fracture_type = Fracture;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;
    double G0;
    double s0;
    double bond_break_coeff;

    ForceModel() {}
    ForceModel( const double _delta, const double _K, const double _G,
                const double _G0, const int _influence = 0 )
        : base_type( _delta, _K, _G, _influence )
    {
        set_param( _delta, _K, _G, _G0 );
    }

    void set_param( const double _delta, const double _K, const double _G,
                    const double _G0 )
    {
        base_type::set_param( _delta, _K, _G );
        G0 = _G0;
        if ( influence_type == 1 )
        {
            s0 = sqrt( 5.0 * G0 / 9.0 / K / delta ); // 1/xi
        }
        else
        {
            s0 = sqrt( 8.0 * G0 / 15.0 / K / delta ); // 1
        }
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }
};

template <>
struct ForceModel<LinearLPS, Elastic> : public ForceModel<LPS, Elastic>
{
    using base_type = ForceModel<LPS, Elastic>;
    using base_model = LPS;
    using fracture_type = Elastic;

    using base_type::base_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;
};

template <>
struct ForceModel<LinearLPS, Fracture> : public ForceModel<LPS, Fracture>
{
    using base_type = ForceModel<LPS, Fracture>;
    using base_model = LPS;
    using fracture_type = Fracture;

    using base_type::base_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;
};

/* PMB */
template <>
struct ForceModel<PMB, Elastic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using base_model = PMB;
    using fracture_type = Elastic;

    using BaseForceModel::delta;

    double c;
    double K;

    ForceModel(){};
    ForceModel( const double delta, const double K )
        : base_type( delta )
    {
        set_param( delta, K );
    }

    ForceModel( const ForceModel& model )
        : base_type( model )
    {
        c = model.c;
        K = model.K;
    }

    void set_param( const double _delta, const double _K )
    {
        delta = _delta;
        K = _K;
        c = 18.0 * K / ( 3.141592653589793 * delta * delta * delta * delta );
    }
};

template <>
struct ForceModel<PMB, Fracture> : public ForceModel<PMB, Elastic>
{
    using base_type = ForceModel<PMB, Elastic>;
    using base_model = PMB;
    using fracture_type = Fracture;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
    double G0;
    double s0;
    double bond_break_coeff;

    ForceModel() {}
    ForceModel( const double delta, const double K, const double G0 )
        : base_type( delta, K )
    {
        set_param( delta, K, G0 );
    }

    ForceModel( const ForceModel& model )
        : base_type( model )
    {
        G0 = model.G0;
        s0 = model.s0;
        bond_break_coeff = model.bond_break_coeff;
    }

    void set_param( const double _delta, const double _K, const double _G0 )
    {
        base_type::set_param( _delta, _K );
        G0 = _G0;
        s0 = sqrt( 5.0 * G0 / 9.0 / K / delta );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }
};

template <>
struct ForceModel<LinearPMB, Elastic> : public ForceModel<PMB, Elastic>
{
    using base_type = ForceModel<PMB, Elastic>;
    using base_model = PMB;
    using fracture_type = Elastic;

    using base_type::base_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
};

template <>
struct ForceModel<LinearPMB, Fracture> : public ForceModel<PMB, Fracture>
{
    using base_type = ForceModel<PMB, Fracture>;
    using base_model = PMB;
    using fracture_type = Fracture;

    using base_type::base_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;
};

/******************************************************************************
  Force helper functions.
******************************************************************************/
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
class Force<ExecutionSpace, ForceModel<LPS, Elastic>>
{
  protected:
    bool _half_neigh;
    ForceModel<LPS, Elastic> _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<LPS, Elastic> model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
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
        auto model = _model;

        auto weighted_volume = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the reference positions and displacements.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            double m_j = model.influence_function( xi ) * xi * xi * vol( j );
            m( i ) += m_j;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, weighted_volume, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::compute_weighted_volume" );
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void compute_dilatation( ParticleType& particles,
                             const NeighListType& neigh_list,
                             const ParallelType neigh_op_tag ) const
    {
        auto n_local = particles.n_local;
        const auto x = particles.slice_x();
        auto u = particles.slice_u();
        const auto vol = particles.slice_vol();
        auto m = particles.slice_m();
        auto theta = particles.slice_theta();
        auto model = _model;
        Cabana::deep_copy( theta, 0.0 );

        auto dilatation = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the bond distance, displacement, and stretch.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            double theta_i =
                model.influence_function( xi ) * s * xi * xi * vol( j );
            theta( i ) += 3.0 * theta_i / m( i );
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, dilatation, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::compute_dilatation" );
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
        auto model = _model;

        const auto vol = particles.slice_vol();
        auto theta = particles.slice_theta();
        auto m = particles.slice_m();
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
                  s_coeff * s * ( 1.0 / m( i ) + 1.0 / m( j ) ) ) *
                model.influence_function( xi ) * xi * vol( j );
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
        auto model = _model;

        const auto vol = particles.slice_vol();
        const auto theta = particles.slice_theta();
        auto m = particles.slice_m();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            double num_neighbors = static_cast<double>(
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i ) );

            double w = ( 1.0 / num_neighbors ) * 0.5 * theta_coeff / 3.0 *
                           ( theta( i ) * theta( i ) ) +
                       0.5 * ( s_coeff / m( i ) ) *
                           model.influence_function( xi ) * s * s * xi * xi *
                           vol( j );
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
class Force<ExecutionSpace, ForceModel<LPS, Fracture>>
    : public Force<ExecutionSpace, ForceModel<LPS, Elastic>>
{
  protected:
    using base_type = Force<ExecutionSpace, ForceModel<LPS, Elastic>>;
    using base_type::_half_neigh;
    ForceModel<LPS, Fracture> _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<LPS, Fracture> model )
        : base_type( half_neigh, model )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class MuView>
    void compute_weighted_volume( ParticleType& particles,
                                  const NeighListType& neigh_list,
                                  const MuView& mu )
    {
        auto n_local = particles.n_local;
        auto x = particles.slice_x();
        auto u = particles.slice_u();
        const auto vol = particles.slice_vol();
        auto m = particles.slice_m();
        Cabana::deep_copy( m, 0.0 );
        auto model = _model;

        auto weighted_volume = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );

                // Get the reference positions and displacements.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );
                // Added mu to account for bond breaking.
                double m_j = mu( i, n ) * model.influence_function( xi ) * xi *
                             xi * vol( j );
                m( i ) += m_j;
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for(
            "CabanaPD::ForceLPSDamage::compute_weighted_volume", policy,
            weighted_volume );
    }

    template <class ParticleType, class NeighListType, class MuView>
    void compute_dilatation( ParticleType& particles,
                             const NeighListType& neigh_list,
                             const MuView& mu ) const
    {
        auto n_local = particles.n_local;
        const auto x = particles.slice_x();
        auto u = particles.slice_u();
        const auto vol = particles.slice_vol();
        auto m = particles.slice_m();
        auto theta = particles.slice_theta();
        auto model = _model;
        Cabana::deep_copy( theta, 0.0 );

        auto dilatation = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );

                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );
                // Added mu to account for bond breaking.
                double theta_i = mu( i, n ) * model.influence_function( xi ) *
                                 s * xi * xi * vol( j );
                // m( i ) is 0 when all bonds connected to particle i are
                // broken.
                if ( m( i ) > 0 )
                    theta( i ) += 3.0 * theta_i / m( i );
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::ForceLPSDamage::compute_dilatation",
                              policy, dilatation );
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    void compute_force_full( ForceType& f, const PosType& x, const PosType& u,
                             const ParticleType& particles,
                             const NeighListType& neigh_list, MuView& mu,
                             const int n_local, ParallelType& ) const
    {
        auto break_coeff = _model.bond_break_coeff;
        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.slice_vol();
        auto theta = particles.slice_theta();
        auto m = particles.slice_m();
        const auto nofail = particles.slice_nofail();
        auto force_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                double fx_i = 0.0;
                double fy_i = 0.0;
                double fz_i = 0.0;

                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );

                // Get the reference positions and displacements.
                double xi, r, s;
                double rx, ry, rz;
                getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

                // Break if beyond critical stretch unless in no-fail zone.
                if ( r * r >= break_coeff * xi * xi && !nofail( i ) &&
                     !nofail( i ) )
                {
                    mu( i, n ) = 0;
                    // Note m( i ) = 0 when all bonds connected to particle i
                    // are broken, which implies mu( i, n ) = 0 for all
                    // neighbors of particle i. Similarly for m( j ).
                }
                else if ( mu( i, n ) > 0 )
                {
                    const double coeff =
                        ( theta_coeff *
                              ( theta( i ) / m( i ) + theta( j ) / m( j ) ) +
                          s_coeff * s * ( 1.0 / m( i ) + 1.0 / m( j ) ) ) *
                        model.influence_function( xi ) * xi * vol( j );
                    double muij = mu( i, n );
                    fx_i = muij * coeff * rx / r;
                    fy_i = muij * coeff * ry / r;
                    fz_i = muij * coeff * rz / r;

                    f( i, 0 ) += fx_i;
                    f( i, 1 ) += fy_i;
                    f( i, 2 ) += fz_i;
                }
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::ForceLPSDamage::compute_full", policy,
                              force_full );
    }

    template <class PosType, class WType, class DamageType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    double compute_energy_full( WType& W, const PosType& x, const PosType& u,
                                DamageType& phi, const ParticleType& particles,
                                const NeighListType& neigh_list, MuView& mu,
                                const int n_local, ParallelType& ) const
    {
        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.slice_vol();
        const auto theta = particles.slice_theta();
        auto m = particles.slice_m();

        auto energy_full = KOKKOS_LAMBDA( const int i, double& Phi )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            double num_bonds = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
                num_bonds += static_cast<double>( mu( i, n ) );

            double phi_i = 0.0;
            double vol_H_i = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );
                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                double w =
                    mu( i, n ) * ( ( 1.0 / num_bonds ) * 0.5 * theta_coeff /
                                       3.0 * ( theta( i ) * theta( i ) ) +
                                   0.5 * ( s_coeff / m( i ) ) *
                                       model.influence_function( xi ) * s * s *
                                       xi * xi * vol( j ) );
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
            "CabanaPD::ForceLPSDamage::compute_energy_full", policy,
            energy_full, strain_energy );

        return strain_energy;
    }
};

template <class ExecutionSpace>
class Force<ExecutionSpace, ForceModel<LinearLPS, Elastic>>
    : public Force<ExecutionSpace, ForceModel<LPS, Elastic>>
{
  protected:
    using base_type = Force<ExecutionSpace, ForceModel<LPS, Elastic>>;
    using base_type::_half_neigh;
    ForceModel<LinearLPS, Elastic> _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<LinearLPS, Elastic> model )
        : base_type( half_neigh, model )
        , _model( model )
    {
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
        auto model = _model;

        const auto vol = particles.slice_vol();
        auto linear_theta = particles.slice_theta();
        // Using weighted volume from base LPS class.
        auto m = particles.slice_m();

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            // Get the bond distance and linearized stretch.
            double xi, linear_s;
            double xi_x, xi_y, xi_z;
            getLinearizedDistanceComponents( x, u, i, j, xi, linear_s, xi_x,
                                             xi_y, xi_z );

            const double coeff =
                ( theta_coeff * ( linear_theta( i ) / m( i ) +
                                  linear_theta( j ) / m( j ) ) +
                  s_coeff * linear_s * ( 1.0 / m( i ) + 1.0 / m( j ) ) ) *
                model.influence_function( xi ) * xi * vol( j );
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
        auto model = _model;

        const auto vol = particles.slice_vol();
        const auto linear_theta = particles.slice_theta();
        auto m = particles.slice_m();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Do we need to recompute linear_theta_i?

            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            double num_neighbors = static_cast<double>(
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i ) );

            double w = ( 1.0 / num_neighbors ) * 0.5 * theta_coeff / 3.0 *
                           ( linear_theta( i ) * linear_theta( i ) ) +
                       0.5 * ( s_coeff / m( i ) ) *
                           model.influence_function( xi ) * linear_s *
                           linear_s * xi * xi * vol( j );
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
class Force<ExecutionSpace, ForceModel<PMB, Elastic>>
{
  protected:
    bool _half_neigh;
    ForceModel<PMB, Elastic> _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<PMB, Elastic> model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void compute_weighted_volume( ParticleType&, const NeighListType&,
                                  const ParallelType )
    {
    }
    template <class ParticleType, class NeighListType, class ParallelType>
    void compute_dilatation( ParticleType&, const NeighListType&,
                             const ParallelType )
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
            // Get the bond distance, displacement, and stretch.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
            // the integrand (pairwise potential).
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
class Force<ExecutionSpace, ForceModel<PMB, Fracture>>
    : public Force<ExecutionSpace, ForceModel<PMB, Elastic>>
{
  protected:
    using base_type = Force<ExecutionSpace, ForceModel<PMB, Elastic>>;
    using base_type::_half_neigh;
    ForceModel<PMB, Fracture> _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<PMB, Fracture> model )
        : base_type( half_neigh, model )
        , _model( model )
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
        const auto nofail = particles.slice_nofail();

        auto force_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                double fx_i = 0.0;
                double fy_i = 0.0;
                double fz_i = 0.0;

                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );

                // Get the reference positions and displacements.
                double xi, r, s;
                double rx, ry, rz;
                getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

                // Break if beyond critical stretch unless in no-fail zone.
                if ( r * r >= break_coeff * xi * xi && !nofail( i ) &&
                     !nofail( j ) )
                {
                    mu( i, n ) = 0;
                }
                else if ( mu( i, n ) > 0 )
                {
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
                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                // 0.25 factor is due to 1/2 from outside the integral and 1/2
                // from the integrand (pairwise potential).
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
class Force<ExecutionSpace, ForceModel<LinearPMB, Elastic>>
    : public Force<ExecutionSpace, ForceModel<PMB, Elastic>>
{
  protected:
    using base_type = Force<ExecutionSpace, ForceModel<PMB, Elastic>>;
    using base_type::_half_neigh;
    ForceModel<LinearPMB, Elastic> _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<LinearPMB, Elastic> model )
        : base_type( half_neigh, model )
        , _model( model )
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

            // Get the bond distance, displacement, and linearized stretch.
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
            // Get the bond distance, displacement, and linearized stretch.
            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
            // the integrand (pairwise potential).
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

    // Forces only atomic if using team threading.
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

    // Forces only atomic if using team threading.
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
