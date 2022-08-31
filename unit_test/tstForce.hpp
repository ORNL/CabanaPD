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

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_config.hpp>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Particles.hpp>

namespace Test
{

//---------------------------------------------------------------------------//
// Get the strain energy density (at one point). Should converge at high enough
// values of m to 9.0 / 2.0 * K * s0 * s0;
double computeReferenceStrainEnergyDensity( const int m, const double delta,
                                            const double K, const double s0 )
{
    double W = 0.0;
    double c = 18.0 * K / ( 3.141592653589793 * delta * delta * delta * delta );
    double dx = delta / m;
    double vol = dx * dx * dx;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi_sq = xi_x * xi_x + xi_y * xi_y + xi_z * xi_z;

                if ( xi_sq > 0.0 && xi_sq < delta * delta + 1e-14 )
                {
                    W += 0.25 * c * s0 * s0 * sqrt( xi_sq ) * vol;
                }
            }

    return W;
}
//---------------------------------------------------------------------------//
template <class ModelType>
void testForce( ModelType model )
{
    using exec_space = TEST_EXECSPACE;
    using mem_space = TEST_MEMSPACE;
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 15, 15, 15 };
    double s0 = 2.0;

    // Create particles based on the mesh.
    using ptype = CabanaPD::Particles<mem_space>;
    ptype particles( exec_space{}, box_min, box_max, num_cells, 0 );
    // This is probably not the best idea - the slice could be changed before
    // this gets called. It certainly has to be called after the particle
    // creation kernel is done.
    auto x = particles.slice_x();
    auto u = particles.slice_u();
    auto v = particles.slice_v();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        for ( int d = 0; d < 3; d++ )
        {
            u( pid, d ) = s0 * x( pid, d );
            v( pid, d ) = 0.0;
        }
    };
    particles.update_particles( exec_space{}, init_functor );

    // This needs to exactly match the mesh spacing to compare with the single
    // particle calculation.
    double delta = ( box_max[0] - box_min[0] ) / num_cells[0];
    double K = 1.0;
    model.set_param( K, delta );
    CabanaPD::Force<exec_space, ModelType> force( true, model );

    double mesh_min[3] = { particles.ghost_mesh_lo[0],
                           particles.ghost_mesh_lo[1],
                           particles.ghost_mesh_lo[2] };
    double mesh_max[3] = { particles.ghost_mesh_hi[0],
                           particles.ghost_mesh_hi[1],
                           particles.ghost_mesh_hi[2] };
    using verlet_list =
        Cabana::VerletList<mem_space, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    // Add to delta to make sure neighbors are found.
    verlet_list neigh_list( x, 0, particles.n_local, delta + 1e-14, 1.0,
                            mesh_min, mesh_max );

    auto f = particles.slice_f();
    auto W = particles.slice_W();
    auto vol = particles.slice_vol();
    compute_force( force, particles, neigh_list, Cabana::SerialOpTag() );

    auto Phi =
        compute_energy( force, particles, neigh_list, Cabana::SerialOpTag() );

    // Make a copy of final results on the host
    std::size_t num_particle = x.size();
    using HostAoSoA =
        Cabana::AoSoA<Cabana::MemberTypes<double[3], double[3], double, double>,
                      Kokkos::HostSpace>;
    HostAoSoA aosoa_host( "host_aosoa", num_particle );
    auto f_host = Cabana::slice<0>( aosoa_host );
    auto x_host = Cabana::slice<1>( aosoa_host );
    auto W_host = Cabana::slice<2>( aosoa_host );
    auto vol_host = Cabana::slice<3>( aosoa_host );
    Cabana::deep_copy( f_host, f );
    Cabana::deep_copy( x_host, x );
    Cabana::deep_copy( W_host, W );
    Cabana::deep_copy( vol_host, vol );

    double ref_W = computeReferenceStrainEnergyDensity( 1, delta, K, s0 );

    // Check the results: avoid the system boundary for per particle values.
    double check_Phi = 0.0;
    for ( std::size_t p = 0; p < num_particle; ++p )
    {
        if ( x_host( p, 0 ) > particles.local_mesh_lo[0] + delta * 1.1 &&
             x_host( p, 0 ) < particles.local_mesh_hi[0] - delta * 1.1 &&
             x_host( p, 1 ) > particles.local_mesh_lo[1] + delta * 1.1 &&
             x_host( p, 1 ) < particles.local_mesh_hi[1] - delta * 1.1 &&
             x_host( p, 2 ) > particles.local_mesh_lo[2] + delta * 1.1 &&
             x_host( p, 2 ) < particles.local_mesh_hi[2] - delta * 1.1 )
        {
            // Check force: all should cancel to zero.
            for ( std::size_t d = 0; d < 3; ++d )
            {
                EXPECT_LE( f_host( p, d ), 1e-13 );
            }

            // Check strain energy (all should be equal for fixed stretch).
            EXPECT_FLOAT_EQ( W_host( p ), ref_W );
        }

        // Check total sum of strain energy matches per particle sum.
        check_Phi += W_host( p ) * vol_host( p );
    }

    EXPECT_FLOAT_EQ( Phi, check_Phi );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_force_pmb ) { testForce( CabanaPD::PMBModel() ); }
TEST( TEST_CATEGORY, test_force_linear_pmb )
{
    testForce( CabanaPD::LinearPMBModel() );
}

} // end namespace Test
