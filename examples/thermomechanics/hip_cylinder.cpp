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

    // Impose separate density values for powder and container particles.
    double W = inputs["wall_thickness"];
    double D0 = inputs["powder_initial_relative_density"];
    auto rho = particles.sliceDensity();
    auto temp = particles.sliceTemperature();
    auto x = particles.sliceReferencePosition();
    auto nofail = particles.sliceNoFail();

    using pool_type = Kokkos::Random_XorShift64_Pool<exec_space>;
    using random_type = Kokkos::Random_XorShift64<exec_space>;
    pool_type pool;
    int seed = 456854;
    pool.init( seed, particles.numLocal() );

    // Use time to seed random number generator
    // std::srand( std::time( nullptr ) );
    double rho_perturb_factor = 0.02;

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        double rsq = ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center ) +
                     ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center );
        if ( rsq > ( Rin + W ) * ( Rin + W ) &&
             rsq < ( Rout - W ) * ( Rout - W ) &&
             x( pid, 2 ) < z_center + 0.5 * H - W &&
             x( pid, 2 ) > z_center - 0.5 * H + W )
        { // Powder density
            rho( pid ) = D0 * rho0;
            // Perturb powder density
            auto gen = pool.get_state();
            auto rand =
                Kokkos::rand<random_type, double>::draw( gen, 0.0, 1.0 );
            double factor = ( 1 + ( 2.0 * rand - 1.0 ) * rho_perturb_factor );

            //  double factor =
            //      ( 1 + ( -1 + 2 * ( (double)std::rand() / ( RAND_MAX ) ) ) *
            //                 rho_perturb_factor );
            rho( pid ) *= factor;
        }
        else
        { // Container density
            rho( pid ) = rho0;
        };
        // Temperature
        temp( pid ) = temp0;
        // No fail
        nofail( pid ) = 1;
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
    auto rho_current = particles.sliceCurrentDensity();
    temp = particles.sliceTemperature();
    CabanaPD::ForceDensityModel force_model(
        model_type{}, mechanics_type{}, rho, rho_current, delta, K, G0, sigma_y,
        rho0, temp, alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );
    /*
    using contact_type = CabanaPD::NormalRepulsionModel;
    auto dx = particles.dx;

    // Use contact radius and extension relative to particle spacing.
    double r_c = inputs["contact_horizon_factor"];
    double r_extend = inputs["contact_horizon_extend_factor"];
    // NOTE: dx/2 is when particles first touch.
    r_c *= dx[0] / 2.0;
    r_extend *= dx[0];

    contact_type contact_model( delta, r_c, r_extend, K );

    CabanaPD::Solver solver( inputs, particles, force_model,
                                     contact_model );
                                     */

    // ====================================================
    //                    Impose field
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double Pmax = inputs["maximum_pressure"];
    double Tmax = inputs["maximum_temperature"];
    double dx = solver.particles.dx[0];
    double dz = solver.particles.dx[2];
    double b0 = Pmax / dx;
    x = solver.particles.sliceReferencePosition();
    auto f = solver.particles.sliceForce();
    auto u = solver.particles.sliceDisplacement();
    temp = solver.particles.sliceTemperature();
    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto force_temp_func = KOKKOS_LAMBDA( const int pid, const double )
    {
        // -----------------------
        //  Isostatic pressure BC
        // -----------------------
        double rsq = ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center ) +
                     ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center );
        double theta =
            Kokkos::atan2( x( pid, 1 ) - y_center, x( pid, 0 ) - x_center );

        // BC on outer boundary
        // if ( rsq > ( Rout - dx ) * ( Rout - dx ) )
        if ( rsq > ( Rout - W ) * ( Rout - W ) )
        {
            f( pid, 0 ) += -b0 * Kokkos::cos( theta );
            f( pid, 1 ) += -b0 * Kokkos::sin( theta );
        }
        // BC on inner boundary
        // else if ( rsq < ( Rin + dx ) * ( Rin + dx ) )
        else if ( rsq < ( Rin + W ) * ( Rin + W ) )
        {
            f( pid, 0 ) += b0 * Kokkos::cos( theta );
            f( pid, 1 ) += b0 * Kokkos::sin( theta );
        };

        // BC on top boundary
        // if ( x( pid, 2 ) > z_center + 0.5 * H - dz )
        if ( x( pid, 2 ) > z_center + 0.5 * H - W )
        {
            f( pid, 2 ) += -b0;
        }
        // BC on bottom boundary
        // else if ( x( pid, 2 ) < z_center - 0.5 * H + dz )
        else if ( x( pid, 2 ) < z_center - 0.5 * H + W )
        {
            f( pid, 2 ) += b0;
        };

        // -----------------------
        //      Temperature BC
        // -----------------------
        temp( pid ) = Tmax;

        // -----------------------
        // Constrain displacements
        // -----------------------
        double Rmid = 0.5 * ( Rin + Rout );

        if ( x( pid, 2 ) > z_center - dz && x( pid, 2 ) < z_center + dz &&
             rsq > ( Rmid - dx ) * ( Rmid - dx ) &&
             rsq < ( Rmid + dx ) * ( Rmid + dx ) )
        {
            u( pid, 0 ) = 0.0;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
    };
    CabanaPD::BodyTerm body_term( force_temp_func, solver.particles.size(),
                                  true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( body_term );
    solver.run( body_term );
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
