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
    double G0 = inputs["fracture_energy"];
    double sigma_y = inputs["yield_stress"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    double alpha = inputs["thermal_expansion_coeff"];

    // Problem parameters
    double temp0 = inputs["reference_temperature"];

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
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // Temperature
        temp( pid ) = temp0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    // auto force_model = CabanaPD::createForceModel(
    //    model_type{}, mechanics_type{}, CabanaPD::Fracture{}, *particles,
    //    delta, K, G0, sigma_y, alpha, temp0 );

    auto force_model =
        CabanaPD::createForceModel( model_type{}, CabanaPD::Fracture{},
                                    *particles, delta, K, G0, alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd =
        CabanaPD::createSolver<memory_space>( inputs, particles, force_model );

    // ====================================================
    //                   Imposed field
    // ====================================================
    const double low_corner_y = low_corner[1];
    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        // temp( pid ) = temp0 + 5000.0 * ( x( pid, 1 ) - low_corner_y ) * t;
        temp( pid ) = temp0;
    };
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
