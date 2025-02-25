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

#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

// Simulate thermo-elastic-plastic deformation in a divertor monoblock due to
// imposed thermal field.
void thermoElasticPlasticMonoblockExample( const std::string filename )
{
    // ====================================================
    //             Use default Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //            Material and problem parameters
    // ====================================================
    // Material parameters
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25;
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double sc = inputs["critical_stretch"];
    // double G0 = inputs["fracture_energy"];
    double sigma_y = inputs["yield_stress"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    double G0 = 9 * K * delta * ( sc * sc ) / 5;
    double alpha = inputs["thermal_expansion_coeff"];

    // Problem parameters
    double temp0 = inputs["reference_temperature"];
    double temp_initial = inputs["initial_temperature"];

    // ====================================================
    //                  Discretization
    // ====================================================
    // FIXME: set halo width based on delta
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                Force model type
    // ====================================================
    using model_type = CabanaPD::PMB;
    using thermal_type = CabanaPD::TemperatureDependent;
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    auto particles =
        CabanaPD::createParticles<memory_space, model_type, thermal_type>(
            exec_space(), low_corner, high_corner, num_cells, halo_width );

    // Do not create particles within given cylindrical region
    auto x = particles->sliceReferencePosition();
    double x_center = inputs["cylindrical_hole"][0];
    double y_center = inputs["cylindrical_hole"][1];
    double radius = inputs["cylindrical_hole"][2];
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        if ( ( ( x[0] - x_center ) * ( x[0] - x_center ) +
               ( x[1] - y_center ) * ( x[1] - y_center ) ) < radius * radius )
            return false;
        return true;
    };

    particles->createParticles( exec_space(), init_op );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles->sliceDensity();
    auto temp = particles->sliceTemperature();
    x = particles->sliceReferencePosition();
    auto dx = particles->dx;
    double factor = inputs["grid_perturbation_factor"];

    using pool_type = Kokkos::Random_XorShift64_Pool<exec_space>;
    using random_type = Kokkos::Random_XorShift64<exec_space>;
    pool_type pool;
    int seed = 456854;
    pool.init( seed, particles->localOffset() );

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // Temperature
        temp( pid ) = temp_initial;

        // Perturb particle positions
        auto gen = pool.get_state();
        for ( std::size_t d = 0; d < 3; d++ )
        {
            auto rand =
                Kokkos::rand<random_type, double>::draw( gen, 0.0, 1.0 );
            x( pid, d ) += ( 2.0 * rand - 1.0 ) * factor * dx[d];
        }
        pool.free_state( gen );
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    // auto force_model =
    //    CabanaPD::createForceModel( model_type{}, CabanaPD::NoFracture{},
    //                                *particles, delta, K, alpha, temp0 );

    // auto force_model =
    //     CabanaPD::createForceModel( model_type{}, CabanaPD::Fracture{},
    //                                  *particles, delta, K, G0, alpha, temp0
    //                                  );

    auto force_model = CabanaPD::createForceModel(
        model_type{}, mechanics_type{}, CabanaPD::Fracture{}, *particles, delta,
        K, G0, sigma_y, alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd =
        CabanaPD::createSolver<memory_space>( inputs, particles, force_model );

    // ====================================================
    //                   Imposed field
    // ====================================================
    x = particles->sliceReferencePosition();
    temp = particles->sliceTemperature();

    double dt_ramp_up_steady_state = inputs["time_ramp_up_steady_state"];
    double dt_ramp_up_shock = inputs["time_ramp_up_shock"];
    double dt_ramp_down_shock = inputs["time_ramp_down_shock"];

    const double low_corner_y = low_corner[1];
    double coolant_tubing_width = inputs["coolant_tubing_width"];
    double y_top_coolant_hole = y_center + radius - coolant_tubing_width;

    double temp_max_steady_state = inputs["max_steady_state_temperature"];
    double temp_max_shock = inputs["max_shock_temperature"];

    double ramp_up_steady_state_rate =
        ( temp_max_steady_state - temp_initial ) /
        ( ( high_corner[1] - y_top_coolant_hole ) * dt_ramp_up_steady_state );
    double ramp_up_shock_rate =
        ( temp_max_shock - temp_initial ) /
        ( ( high_corner[1] - y_top_coolant_hole ) * dt_ramp_up_shock );
    double ramp_down_shock_rate =
        ( temp_max_shock - temp_initial ) /
        ( ( high_corner[1] - y_top_coolant_hole ) * dt_ramp_down_shock );

    double pulse_width = inputs["pulse_width"];

    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        if ( x( pid, 1 ) > y_top_coolant_hole )
        {
            // --------------------------------------------
            //           Steady-state profile
            // --------------------------------------------
            if ( t < dt_ramp_up_steady_state )
            {
                temp( pid ) = temp_initial;
                temp( pid ) += ramp_up_steady_state_rate *
                               ( x( pid, 1 ) - y_top_coolant_hole ) * t;
            }
            // --------------------------------------------
            //       Thermal shock profile (ramp up)
            // --------------------------------------------
            else if ( t < dt_ramp_up_steady_state + dt_ramp_up_shock )
            {
                if ( Kokkos::abs( x( pid, 0 ) - x_center ) < 0.5 * pulse_width )
                {
                    temp( pid ) = temp_initial;
                    temp( pid ) += ramp_up_steady_state_rate *
                                   ( x( pid, 1 ) - y_top_coolant_hole ) *
                                   dt_ramp_up_steady_state;
                    temp( pid ) += ramp_up_shock_rate *
                                   ( x( pid, 1 ) - y_top_coolant_hole ) *
                                   ( t - dt_ramp_up_steady_state );
                };
            }
            // --------------------------------------------
            //       Thermal shock profile (ramp down)
            // --------------------------------------------
            else if ( t < dt_ramp_up_steady_state + dt_ramp_up_shock +
                              dt_ramp_down_shock )
            {
                if ( Kokkos::abs( x( pid, 0 ) - x_center ) < 0.5 * pulse_width )
                {
                    temp( pid ) = temp_initial;
                    temp( pid ) += ramp_up_steady_state_rate *
                                   ( x( pid, 1 ) - y_top_coolant_hole ) *
                                   dt_ramp_up_steady_state;
                    temp( pid ) += ramp_up_shock_rate *
                                   ( x( pid, 1 ) - y_top_coolant_hole ) *
                                   dt_ramp_up_shock;
                    temp( pid ) -=
                        ramp_down_shock_rate *
                        ( x( pid, 1 ) - y_top_coolant_hole ) *
                        ( t - ( dt_ramp_up_steady_state + dt_ramp_up_shock ) );
                };
            }
            // --------------------------------------------
            //             End of shock
            // --------------------------------------------
            else
            {
                temp( pid ) = temp_initial;
                temp( pid ) += ramp_up_steady_state_rate *
                               ( x( pid, 1 ) - y_top_coolant_hole ) *
                               dt_ramp_up_steady_state;
            };
        };
    };

    // --------------------------------------------
    //                Thermal shock
    // --------------------------------------------

    auto body_term = CabanaPD::createBodyTerm( temp_func, false );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( body_term );
    cabana_pd->run( body_term );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    thermoElasticPlasticMonoblockExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
