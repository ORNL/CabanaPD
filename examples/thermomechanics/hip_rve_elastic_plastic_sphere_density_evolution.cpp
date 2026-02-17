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

// Simulate a spherical representative volume element (RVE) under isostatic
// pressing with an elastic-perfectly plastic model.
void IPrveElasticPerfectlyPlasticExample( const std::string filename )
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
    double reduced_rho_factor = inputs["reduced_density_factor"];
    rho0 *= reduced_rho_factor; // Using an initial reduced density
    double E = inputs["elastic_modulus"];
    double nu = 0.25; // Use bond-based model
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double sigma_y = inputs["yield_stress"];
    double G0 = inputs["fracture_energy"];

    double delta = inputs["horizon"];
    delta += 1e-10;

    double alpha = inputs["thermal_expansion_coeff"];
    double kappa = inputs["thermal_conductivity"];
    double cp = inputs["specific_heat_capacity"];

    // Problem parameters
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
    //                 Force model type
    // ====================================================
    using model_type = CabanaPD::PMB;
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;
    using thermal_type = CabanaPD::TemperatureDependent;

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double x_size = inputs["system_size"][0];
    double R = x_size / 2.0;
    double R2 = R * R;

    // Do not create particles outside given region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        if ( r2 > R2 )
            return false;
        return true;
    };

    CabanaPD::Particles particles( memory_space{}, model_type{}, thermal_type{},
                                   CabanaPD::EnergyOutput{},
                                   CabanaPD::DynamicDensity{} );
    particles.domain( low_corner, high_corner, num_cells, halo_width );
    particles.create( exec_space{}, init_op );

    auto rho = particles.sliceDensity();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Initial density
        rho( pid ) = rho0;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    rho = particles.sliceDensity();
    auto rho_current = particles.sliceCurrentDensity();
    auto temp = particles.sliceTemperature();
    double contact_r = 0.0;
    CabanaPD::ForceDensityModel force_model(
        model_type{}, CabanaPD::ElasticPerfectlyPlastic{}, rho, rho_current,
        delta, K, G0, sigma_y, rho0, contact_r, temp, kappa, cp, alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                    Impose field
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double W = inputs["wall_thickness"];
    double RW = ( R - W );
    double RW2 = RW * RW;
    double Pmax = inputs["maximum_pressure"];
    double trampup = inputs["ramp_up_bc_time"];
    double trampdown = inputs["ramp_down_bc_time"];
    double tf = inputs["final_time"];

    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );
    double z_center = 0.5 * ( low_corner[2] + high_corner[2] );
    double dx = solver.particles.dx[0];
    double dy = solver.particles.dx[1];
    double dz = solver.particles.dx[2];

    double b0max = Pmax / W;

    auto x = solver.particles.sliceReferencePosition();
    auto u = solver.particles.sliceDisplacement();
    auto f = solver.particles.sliceForce();

    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto bc_func =
        KOKKOS_LAMBDA( const int pid, const double t, const bool, const bool )
    {
        double b0;
        // Pressure ramping
        // Linear profile: f(x) = f(a) + (x-a) * (f(b)-f(a))/(b-a) for x in
        // [a,b]
        if ( t < trampup )
        {
            b0 = t * b0max / trampup;
        }
        else if ( t > tf - trampdown )
        {
            b0 = b0max - ( t - ( tf - trampdown ) ) * b0max / trampdown;
        }
        else
        {
            b0 = b0max;
        }

        const double r2 = x( pid, 0 ) * x( pid, 0 ) +
                          x( pid, 1 ) * x( pid, 1 ) + x( pid, 2 ) * x( pid, 2 );

        // Isostatic pressure
        if ( r2 > RW2 )
        {
            for ( int d = 0; d < 3; d++ )
                f( pid, d ) += -b0 * x( pid, d ) / Kokkos::sqrt( r2 );
        }

        // Constraint 1: fix x-displacement on YZ-plane within BC region
        if ( r2 > RW2 && x( pid, 0 ) > x_center - dx &&
             x( pid, 0 ) < x_center + dx )
        {
            u( pid, 0 ) = 0.0;
        }

        // Constraint 2: fix y-displacement on XZ-plane within BC region
        if ( r2 > RW2 && x( pid, 1 ) > y_center - dy &&
             x( pid, 1 ) < y_center + dy )
        {
            u( pid, 1 ) = 0.0;
        }

        // Constraint 3: fix z-displacement on XY-plane within BC region
        if ( r2 > RW2 && x( pid, 2 ) > z_center - dz &&
             x( pid, 2 ) < z_center + dz )
        {
            u( pid, 2 ) = 0.0;
        }
    };
    CabanaPD::BodyTerm bc( bc_func, solver.particles.size(), true, false );

    // ====================================================
    //                      Outputs
    // ====================================================
    CabanaPD::Region<CabanaPD::SphericalShell> inner_sphere(
        0, RW, x_center, y_center, z_center );

    // Output distance from center in inner sphere.
    auto y = solver.particles.sliceCurrentPosition();
    auto distance_func = KOKKOS_LAMBDA( const int p )
    {
        return Kokkos::sqrt( y( p, 0 ) * y( p, 0 ) + y( p, 1 ) * y( p, 1 ) +
                             y( p, 2 ) * y( p, 2 ) );
    };
    auto output_r = CabanaPD::createOutputTimeSeries(
        "output_distance.txt", inputs, exec_space{}, solver.particles,
        distance_func, inner_sphere );

    // Output average total density.
    auto rho_c = solver.particles.sliceCurrentDensity();
    auto density_func = KOKKOS_LAMBDA( const int p ) { return rho_c( p ); };
    auto output_rho = CabanaPD::createOutputTimeSeries(
        "output_density.txt", inputs, exec_space{}, solver.particles,
        density_func, inner_sphere );

    // Output average total density.
    auto theta_p = solver.particles.slicePlasticDilatation();
    auto plastic_dilatation_func = KOKKOS_LAMBDA( const int p )
    {
        return theta_p( p );
    };
    auto output_theta_p = CabanaPD::createOutputTimeSeries(
        "output_plastic_dilatation.txt", inputs, exec_space{}, solver.particles,
        plastic_dilatation_func, inner_sphere );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc, output_r, output_rho, output_theta_p );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    IPrveElasticPerfectlyPlasticExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
