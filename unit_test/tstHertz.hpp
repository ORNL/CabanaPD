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
template <class VelType, class DensityType, class VolumeType>
double calculateKE( const VelType& v, const DensityType& rho,
                    const VolumeType& vol )
{
    using Kokkos::hypot;
    using Kokkos::pow;

    using execution_space = typename VelType::execution_space;

    double tke;
    Kokkos::parallel_reduce(
        "total_ke", Kokkos::RangePolicy<execution_space>( 0, v.size() ),
        KOKKOS_LAMBDA( const int& i, double& sum ) {
            sum += 0.5 * rho( i ) * vol( i ) *
                   pow( hypot( v( i, 0 ), v( i, 1 ), v( i, 2 ) ), 2.0 );
        },
        Kokkos::Sum<double>( tke ) );

    return tke;
}

void testHertzianContact( const std::string filename )
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
                position( p, 0 ) = 5.1e-5;
            else
                position( p, 0 ) = -5.1e-5;
            volume( p ) = vol;
        } );

    // ====================================================
    //            Force model
    // ====================================================
    using model_type = CabanaPD::HertzianModel;
    // No search radius extension.
    model_type contact_model( radius, radius_extend, nu, E, e );

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

    // Get initial total KE
    double ke_i = calculateKE( v, rho, vo );

    // ====================================================
    //  Simulation run
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, contact_model );
    solver.init();
    solver.run();

    // Get final total KE
    double ke_f = calculateKE( v, rho, vo );

    EXPECT_NEAR( std::sqrt( ke_f / ke_i ), e, 1e-3 );
}

// Test construction.
TEST( TEST_CATEGORY, test_force_hertz_construct )
{
    double radius = 5.0;
    double extend = 1.0;
    double nu = 2.0;
    double E = 100.0;
    double e = 1.0;
    CabanaPD::HertzianModel contact_model( radius, extend, nu, E, e );
}

TEST( TEST_CATEGORY, test_hertzian_contact )
{
    std::string input = "hertzian_contact.json";
    testHertzianContact( input );
}

} // end namespace Test
