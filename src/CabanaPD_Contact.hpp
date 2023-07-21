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

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

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
    // PD horizon
    // Contact radius
    ContactModel( const double _delta, const double _Rc )
        : delta( _delta )
        , Rc( _Rc ){};
};

/* Normal repulsion */

struct NormalRepulsionModel : public ContactModel
{
    using ContactModel::delta;
    using ContactModel::Rc;

    double c;
    double K;

    NormalRepulsionModel(){};
    NormalRepulsionModel( const double delta, const double Rc, const double _K )
        : ContactModel( delta, Rc )
        , K( _K )
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

template <class MemorySpace, class ContactType>
class Contact;

/******************************************************************************
  Normal repulsion computation
******************************************************************************/
template <class MemorySpace>
class Contact<MemorySpace, NormalRepulsionModel>
{
  public:
    using neighbor_type =
        Cabana::VerletList<MemorySpace, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;

    template <class ParticleType>
    Contact( const bool half_neigh, const NormalRepulsionModel model,
             ParticleType& particles )
        : _half_neigh( half_neigh )
        , _model( model )
    {
        // Create contact neighbor list
        for ( std::size_t d = 0; d < 3; d++ )
        {
            mesh_min[d] = particles.ghost_mesh_lo[d];
            mesh_max[d] = particles.ghost_mesh_hi[d];
        }
        const auto y = particles.sliceCurrentPosition();
        _neighbors = std::make_shared<neighbor_type>(
            y, 0, particles.n_local, model.Rc, 1.0, mesh_min, mesh_max );
    }

    template <class ForceType, class ParticleType, class ParallelType>
    void computeContactFull( ForceType& fc, const ParticleType& particles,
                             const int n_local,
                             ParallelType& neigh_op_tag ) const
    {
        auto delta = _model.delta;
        auto Rc = _model.Rc;
        auto c = _model.c;
        const auto vol = particles.sliceVolume();
        const auto x = particles.sliceReferencePosition();
        const auto u = particles.sliceDisplacement();
        const auto y = particles.sliceCurrentPosition();

        _neighbors->build( y, 0, particles.n_local, Rc, 1.0, mesh_min,
                           mesh_max );

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fcx_i = 0.0;
            double fcy_i = 0.0;
            double fcz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

            // Contact "stretch"
            const double sc = ( r - Rc ) / delta;

            // Normal repulsion uses a 15 factor compared to the PMB force
            const double coeff = 15 * c * sc * vol( j );
            fcx_i = coeff * rx / r;
            fcy_i = coeff * ry / r;
            fcz_i = coeff * rz / r;

            fc( i, 0 ) += fcx_i;
            fc( i, 1 ) += fcy_i;
            fc( i, 2 ) += fcz_i;
        };

        // FIXME: using default space for now.
        using exec_space = typename MemorySpace::execution_space;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, contact_full, *_neighbors, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::Contact::compute_full" );
    }

  protected:
    bool _half_neigh;
    NormalRepulsionModel _model;
    std::shared_ptr<neighbor_type> _neighbors;

    double mesh_max[3];
    double mesh_min[3];
};

template <class ForceType, class ParticleType, class ParallelType>
void computeContact( const ForceType& force, ParticleType& particles,
                     const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto f = particles.sliceForce();
    auto f_a = particles.sliceForceAtomic();

    // Do NOT reset force - this is assumed to be done by the primary force
    // kernel.

    // if ( half_neigh )
    // Forces must be atomic for half list
    // compute_force_half( f_a, x, u, neigh_list, n_local,
    //                    neigh_op_tag );

    // Forces only atomic if using team threading
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        force.computeContactFull( f_a, particles, n_local, neigh_op_tag );
    else
        force.computeContactFull( f, particles, n_local, neigh_op_tag );
    Kokkos::fence();
}

} // namespace CabanaPD

#endif
