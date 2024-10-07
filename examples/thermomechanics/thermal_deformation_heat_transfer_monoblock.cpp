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

// Simulate heat transfer in a divertor monoblock geometry.
void thermalDeformationHeatTransferExample( const std::string filename )
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
    double delta = inputs["horizon"];
    double alpha = inputs["thermal_expansion_coeff"];
    double kappa = inputs["thermal_conductivity"];
    double cp = inputs["specific_heat_capacity"];

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
    using thermal_type = CabanaPD::DynamicTemperature;

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    auto particles =
        std::make_shared<CabanaPD::Particles<memory_space, model_type,
                                             typename thermal_type::base_type>>(
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
    auto force_model = CabanaPD::createForceModel(
        model_type{}, CabanaPD::Elastic{}, *particles, delta, K, kappa, cp,
        alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
        inputs, particles, force_model );

    // ====================================================
    //                   Boundary condition
    // ====================================================
    double dx = particles->dx[0];
    double dy = particles->dx[1];
    using plane_type = CabanaPD::RegionBoundary<CabanaPD::RectangularPrism>;
    //  Need to reslice to include ghosted particles on the boundary.
    temp = particles->sliceTemperature();

    plane_type plane( low_corner[0], high_corner[0], low_corner[1],
                      high_corner[1], low_corner[2], high_corner[2] );
    std::vector<plane_type> planes = { plane };

    /*
        auto bc_op = KOKKOS_LAMBDA( const int pid, const double t )
        {
            double rsq = ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center )
       + ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center );

            if ( x( pid, 1 ) >= high_corner[1] - dy && x( pid, 1 ) <=
       high_corner[1] + dy )
            {
                temp( pid ) = temp0 + ( 10000.0 - temp0 ) * t;
            }
            else if ( rsq <= ( radius + dx ) * ( radius + dx ) )
            {
                temp( pid ) = temp0;
            }
        };
        auto bc = createBoundaryCondition( bc_op, exec_space{}, *particles,
       planes, true );

    */

    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        double rsq = ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center ) +
                     ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center );

        if ( x( pid, 1 ) >= high_corner[1] - dy &&
             x( pid, 1 ) <= high_corner[1] + dy )
        {
            // temp( pid ) = temp0 + ( 10000.0 - temp0 ) * t;
            temp( pid ) = 2.0 * temp0;
        }
        else if ( rsq <= ( radius + dx ) * ( radius + dx ) )
        {
            temp( pid ) = 1.5 * temp0;
        }
    };
    auto bc = CabanaPD::createBoundaryCondition(
        temp_func, exec_space{}, *particles, planes, false, 1.0 );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( bc );
    cabana_pd->run( bc );

    // ====================================================
    //                      Outputs
    // ====================================================

    // Output temperature along the y-axis
    auto temp_output = particles->sliceTemperature();
    auto value = KOKKOS_LAMBDA( const int pid ) { return temp_output( pid ); };

    int profile_dim = 1;
    std::string file_name = "temperature_yaxis_profile.txt";
    createOutputProfile( MPI_COMM_WORLD, num_cells[1], profile_dim, file_name,
                         *particles, value );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    thermalDeformationHeatTransferExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
