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

#ifndef CONTACT_H
#define CONTACT_H

#include <cmath>

#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Force.hpp>

namespace CabanaPD
{
/******************************************************************************
  Contact model
******************************************************************************/
struct ContactModel
{
    double delta;
    double Rc;

    ContactModel(){};
    ContactModel( const double _delta, const double _Rc )
        : delta( _delta ){},  // PD horizon
         Rc( _Rc ){};         // Contact radius
};

/* Normal repulsion */

struct NormalRepulsionModel : public ContactModel
{
    using ContactModel::delta;
    using ContactModel::Rc;

    double c;
    double K;

    NormalRepulsionModel(){};
    NormalRepulsionModel( const double delta, const double Rc, const double K )
        : ContactModel( delta ), ContactModel( Rc )
    {
        set_param( delta, Rc, K );
    }

    void set_param( const double _delta, const double _Rc, const double _K )
    {
        delta = _delta;
        Rc = _Rc;
        K = _K;
        // This could inherit from PMB (same c)
        c = 18.0 * K / ( 3.141592653589793 * delta * delta * delta * delta );
    }
};

template <class ExecutionSpace, class ContactType>
class Contact;

/******************************************************************************
  Normal repulsion computation
******************************************************************************/
template <class ExecutionSpace>
class Contact<ExecutionSpace, NormalRepulsionModel>
{
  public:
    using exec_space = ExecutionSpace;
    using neighbor_type =
        Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;

    Contact( const bool half_neigh, const NormalRepulsionModel model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
        // Create contact neighbor list
        double mesh_min[3] = { particles->ghost_mesh_lo[0],
                               particles->ghost_mesh_lo[1],
                               particles->ghost_mesh_lo[2] };
        double mesh_max[3] = { particles->ghost_mesh_hi[0],
                               particles->ghost_mesh_hi[1],
                               particles->ghost_mesh_hi[2] };
        auto x = particles->slice_x();
        auto u = particles->slice_u();
        auto y = particles->slice_y();
        _contact_neighbors = std::make_shared<neighbor_type>( y, 0, particles->n_local,
                                                     contact_model.Rc, 1.0,
                                                     mesh_min, mesh_max );
    }

    template <class ContactType, class PosType, class ParticleType,
              class ParallelType>
    void compute_contact_full( ContactType& fc, const PosType& x, const PosType& u,
                             const ParticleType& particles,
                             const int n_local,
                             ParallelType& neigh_op_tag ) const
    {
        auto delta = _model.delta;
        auto Rc = _model.Rc;
        auto c = _model.c;
        const auto vol = particles.slice_vol();

        _contact_neighbors.build( y, 0, particles->n_local,contact_model.Rc, 1.0,
                                  mesh_min, mesh_max );

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fcx_i = 0.0;
            double fcy_i = 0.0;
            double fcz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

            // Contact "stretch"
            const double sc = (r - Rc)/delta;

            // Normal repulsion uses a 15 factor compared to the PMB force
            const double coeff = 15 * c * sc * vol( j );
            fcx_i = coeff * rx / r; 
            fcy_i = coeff * ry / r;
            fcz_i = coeff * rz / r;

            fc( i, 0 ) += fcx_i;
            fc( i, 1 ) += fcy_i;
            fc( i, 2 ) += fcz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, contact_full, _contact_neighbors, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::Contact::compute_full" );
    }
    
  protected:
    bool _half_neigh;
    NormalRepulsionModel _model;
    neighbor_type _contact_neighbors;

};

// NEED Half force computation

/*
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
*/

} // namespace CabanaPD

#endif