/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
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

// Simulate a cylinder under hot isostatic pressing (HIP).
void HIPCylinderExample( const std::string filename )
{
    // ====================================================
    //               Choose Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                Material parameters
    // ====================================================
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25; // Use bond-based model
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double sigma_y = inputs["yield_stress"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    // For PMB or LPS with influence_type == 1
    // double G0 = 9 * K * delta * ( sc * sc ) / 5;
    // For LPS with influence_type == 0 (default)
    // double G0 = 15 * K * delta * ( sc * sc ) / 8;
    double alpha = inputs["thermal_expansion_coeff"];
    double temp0 = inputs["reference_temperature"];
    double sigma0 = inputs["traction"];

    // ====================================================
    //                  Discretization
    // ====================================================
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
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;
    using thermal_type = CabanaPD::TemperatureDependent;
    using density_type = CabanaPD::DynamicDensity;

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );
    double z_center = 0.5 * ( low_corner[2] + high_corner[2] );
    double Rout = inputs["cylinder_outer_radius"];
    double Rin = inputs["cylinder_inner_radius"];
    double H = inputs["cylinder_height"];

    // Do not create particles outside given cylindrical region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = ( x[0] - x_center ) * ( x[0] - x_center ) +
                     ( x[1] - y_center ) * ( x[1] - y_center );
        if ( rsq < Rin * Rin || rsq > Rout * Rout ||
             x[2] > z_center + 0.5 * H || x[2] < z_center - 0.5 * H )
            return false;
        return true;
    };

    CabanaPD::Particles particles(
        memory_space{}, model_type{}, thermal_type{}, density_type{},
        low_corner, high_corner, num_cells, halo_width, init_op, exec_space{} );

    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    double W = inputs["wall_thickness"];

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        double rsq = ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center ) +
                     ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center );
        if ( rsq > ( Rin + W ) * ( Rin + W ) &&
             rsq > ( Rout - W ) * ( Rout - W ) &&
             x( pid, 2 ) > low_corner[2] + W )
        { // Powder density
            rho( pid ) = 0.7 * rho0;
        }
        else
        { // Container density
            rho( pid ) = rho0;
        };
    };
    particles.updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    CabanaPD::RegionBoundary<CabanaPD::RectangularPrism> plane(
        low_corner[0], high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

    // ====================================================
    //                    Force model
    // ====================================================
    rho = particles.sliceDensity();
    auto temp = particles.sliceTemperature();
    CabanaPD::ForceDensityModel force_model( model_type{}, mechanics_type{},
                                             rho, delta, K, G0, sigma_y, rho0,
                                             temp, alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                Boundary conditions
    // ====================================================

    // Create BC last to ensure ghost particles are included.
    double dx = particles.dx[0];
    // double dy = particles.dx[1];
    double dz = particles.dx[2];
    // double sigma0 = inputs["traction"];
    double b0 = sigma0 / dx;
    auto f = particles.sliceForce();
    x = particles.sliceReferencePosition();
    // Create an isostatic pressure BC.
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {
        double rsq = ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center ) +
                     ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center );
        double theta =
            Kokkos::atan2( x( pid, 1 ) - y_center, x( pid, 0 ) - x_center );

        // BC on outer boundary
        if ( rsq > ( Rout - dx ) * ( Rout - dx ) )
        {
            f( pid, 0 ) += -b0 * Kokkos::cos( theta );
            f( pid, 1 ) += -b0 * Kokkos::sin( theta );
        }
        // BC on inner boundary
        else if ( rsq < ( Rin + dx ) * ( Rin + dx ) )
        {
            f( pid, 0 ) += b0 * Kokkos::cos( theta );
            f( pid, 1 ) += b0 * Kokkos::sin( theta );
        };

        // BC on top boundary
        if ( x( pid, 2 ) > z_center + 0.5 * H - dz )
        {
            f( pid, 2 ) += -b0;
        }
        // BC on bottom boundary
        else if ( x( pid, 2 ) < z_center - 0.5 * H + dz )
        {
            f( pid, 2 ) += b0;
        };
    };
    auto bc =
        createBoundaryCondition( bc_op, exec_space{}, particles, true, plane );

    // ====================================================
    //                   Imposed field
    // ====================================================
    /*
    x = particles.sliceReferencePosition();
    auto temp = particles.sliceTemperature();
    const double low_corner_y = low_corner[1];
    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        temp( pid ) = temp0 + 5000.0 * ( x( pid, 1 ) - low_corner_y ) * t;
    };
    CabanaPD::BodyTerm body_term( temp_func, particles.size(), false );
    */

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc );
    // solver.init( body_term );
    // solver.run( body_term );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    HIPCylinderExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
