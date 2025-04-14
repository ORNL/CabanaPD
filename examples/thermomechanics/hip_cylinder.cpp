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
    double K = inputs["bulk_modulus"];
    // double G = inputs["shear_modulus"]; // Only for LPS.
    double sc = inputs["critical_stretch"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    // For PMB or LPS with influence_type == 1
    double G0 = 9 * K * delta * ( sc * sc ) / 5;
    // For LPS with influence_type == 0 (default)
    // double G0 = 15 * K * delta * ( sc * sc ) / 8;
    double sigma_y = 10.0;
    double alpha = inputs["thermal_expansion_coeff"];
    double temp0 = inputs["reference_temperature"];

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
    //                    Force model
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
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();

    double vrmax = inputs["max_radial_velocity"];
    double vrmin = inputs["min_radial_velocity"];
    double vzmax = inputs["max_vertical_velocity"];
    double zmin = z_center - 0.5 * H;

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // Velocity
        double zfactor = ( ( x( pid, 2 ) - zmin ) / ( 0.5 * H ) ) - 1;
        double vr = vrmax - vrmin * zfactor * zfactor;
        v( pid, 0 ) =
            vr * Kokkos::cos( Kokkos::atan2( x( pid, 1 ), x( pid, 0 ) ) );
        v( pid, 1 ) =
            vr * Kokkos::sin( Kokkos::atan2( x( pid, 1 ), x( pid, 0 ) ) );
        v( pid, 2 ) = vzmax * zfactor;
    };
    particles.updateParticles( exec_space{}, init_functor );

    rho = particles.sliceDensity();
    auto temp = particles.sliceTemperature();
    CabanaPD::ForceDensityModel force_model( model_type{}, mechanics_type{},
                                             rho, delta, K, G0, sigma_y, rho0,
                                             temp, alpha, temp0 );

    CabanaPD::Solver solver( inputs, particles, force_model );
    solver.init();
    solver.run();
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
