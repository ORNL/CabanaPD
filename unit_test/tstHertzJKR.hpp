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

#include <CabanaPD.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace Test
{
template <class ForceType>
double getPulloffForce( const ForceType& f, double vol )
{
    using execution_space = typename ForceType::execution_space;

    double pulloff_force;
    auto min_po = Kokkos::Min<double>( pulloff_force );

    Kokkos::parallel_reduce(
        "pulloff_force", Kokkos::RangePolicy<execution_space>( 0, f.size() ),
        KOKKOS_LAMBDA( const int t, double& min ) {
            min_po.join( min, f( t ) );
        },
        min_po );

    return pulloff_force * vol;
}

void testHertzianJKRContact( const std::string filename )
{
    // ====================================================
    //             Use test Kokkos spaces
    // ====================================================
    using exec_space = TEST_EXECSPACE;
    using memory_space = TEST_MEMSPACE;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                Material parameters
    // ====================================================
    double rho0 = inputs["density"];
    double vol = inputs["volume"];
    double radius = inputs["radius"];
    double radius_extend = inputs["radius_extend"];
    double nu = inputs["poisson_ratio"];
    double E = inputs["elastic_modulus"];
    double e = inputs["restitution"];
    double gamma = inputs["surface_adhesion"];

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];

    // ====================================================
    //            Custom particle creation
    // ====================================================
    const int num_particles = 2;
    // Purposely using zero-init here.
    Kokkos::View<double* [3], memory_space> position( "custom_position", 2 );
    Kokkos::View<double*, memory_space> volume( "custom_volume", 2 );

    Kokkos::parallel_for(
        "create_particles", Kokkos::RangePolicy<exec_space>( 0, num_particles ),
        KOKKOS_LAMBDA( const int p ) {
            if ( p == 0 )
                position( p, 0 ) = 5.01e-5;
            else
                position( p, 0 ) = -5.01e-5;
            volume( p ) = vol;
        } );

    // ====================================================
    //            Force model
    // ====================================================
    using model_type = CabanaPD::HertzianJKRModel;
    // No search radius extension.
    model_type contact_model( radius, radius_extend, nu, E, e, gamma );

    // ====================================================
    //                 Particle generation
    // ====================================================
    int halo_width = 1;
    CabanaPD::Particles particles( memory_space{}, model_type{} );
    particles.domain( low_corner, high_corner, num_cells, halo_width );
    particles.create( exec_space{}, position, volume );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto v = particles.sliceVelocity();
    auto vo = particles.sliceVolume();

    auto init_functor = KOKKOS_LAMBDA( const int p )
    {
        // Density
        rho( p ) = rho0;
        if ( p == 0 )
            v( p, 0 ) = -1.0;
        else
            v( p, 0 ) = 1.0;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //  Simulation run
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, contact_model );
    solver.init();

    Kokkos::View<double*, memory_space> force_time( "forces",
                                                    solver.num_steps );

    auto force = particles.sliceForce();
    for ( int step = 1; step <= solver.num_steps; ++step )
    {
        solver.runStep( step );

        Kokkos::parallel_for(
            "extract_force", Kokkos::RangePolicy<exec_space>( 0, 1 ),
            KOKKOS_LAMBDA( const int ) {
                force_time( step - 1 ) = force( 0, 0 );
            } );
    }

    double min_po = getPulloffForce( force_time, vol );
    double min_po_a = contact_model.fc;

    EXPECT_NEAR( min_po / min_po_a, -1.0, 5e-3 );

    // TODO: We should also test with some amount of damping enabled, similar to
    // the plain Hertz unit test.
}

// Test construction.
TEST( TEST_CATEGORY, test_force_jkr_construct )
{
    double radius = 5.0;
    double extend = 1.0;
    double nu = 2.0;
    double E = 100.0;
    double e = 1.0;
    double gamma = 1.0;
    CabanaPD::HertzianJKRModel contact_model( radius, extend, nu, E, e, gamma );
}

TEST( TEST_CATEGORY, test_hertzian_jkr_contact )
{
    std::string input = "hertzian_jkr_contact.json";
    testHertzianJKRContact( input );
}

} // end namespace Test
