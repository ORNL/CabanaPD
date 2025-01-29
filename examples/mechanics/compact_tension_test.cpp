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

// Simulate crack propagation in a compact tension test.
void compactTensionTestExample( const std::string filename )
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
    //                    Pre-notch
    // ====================================================
    double height = inputs["system_size"][0];
    double thickness = inputs["system_size"][2];
    double L_prenotch = height / 2.0;
    double y_prenotch = 0.0;
    Kokkos::Array<double, 3> p0 = { low_corner[0], y_prenotch, low_corner[2] };
    Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
    Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
    Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_position = { p0 };
    CabanaPD::Prenotch<1> prenotch( v1, v2, notch_position );

    // ====================================================
    //                Force model type
    // ====================================================
    using model_type = CabanaPD::PMB;
    using thermal_type = CabanaPD::TemperatureIndependent;
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    // Geometric parameters of specimen
    double L = inputs["system_size"][1];
    double W = L / 1.25;
    double a = 0.45 * W;

    // Grid spacing in y-direction
    double dy = inputs["dx"][1];

    // Do not create particles outside compact tension test specimen region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Thin rectangle
        if ( x[0] < low_corner[1] + 0.25 * W + a &&
             Kokkos::abs( x[1] ) < 0.5 * dy )
        {
            return false;
        }
        // Thick rectangle
        else if ( x[0] < low_corner[1] + 0.25 * W &&
                  Kokkos::abs( x[1] ) < 26e-4 )
        {
            return false;
        }
        else
        {
            return true;
        }
    };

    auto particles =
        CabanaPD::createParticles<memory_space, model_type, thermal_type>(
            exec_space(), low_corner, high_corner, num_cells, halo_width,
            init_op );

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto v = particles->sliceVelocity();
    auto nofail = particles->sliceNoFail();

    // Pin radius
    double R = 4e-3;
    // Pin center coordinates (top)
    double x_pin = low_corner[0] + 0.25 * W;
    double y_pin = 0.37 * W;
    // Pin velocity magnitude
    double v0 = inputs["pin_velocity"];

    // Create region for each pin.
    CabanaPD::RegionBoundary<CabanaPD::Cylinder> cylinder1(
        0.0, R, low_corner[2], high_corner[2], x_pin, y_pin );
    CabanaPD::RegionBoundary<CabanaPD::Cylinder> cylinder2(
        0.0, R, low_corner[2], high_corner[2], x_pin, -y_pin );
    // Create regions only for setting no-fail condition.
    CabanaPD::RegionBoundary<CabanaPD::Cylinder> nofail_cylinder1(
        0.0, 2.0 * R, low_corner[2], high_corner[2], x_pin, y_pin );
    CabanaPD::RegionBoundary<CabanaPD::Cylinder> nofail_cylinder2(
        0.0, 2.0 * R, low_corner[2], high_corner[2], x_pin, -y_pin );
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // pins' y-velocity
        if ( cylinder1.inside( x, pid ) )
            v( pid, 1 ) = v0;
        else if ( cylinder2.inside( x, pid ) )
            v( pid, 1 ) = -v0;

        // No-fail zone
        if ( nofail_cylinder1.inside( x, pid ) ||
             nofail_cylinder2.inside( x, pid ) )
            nofail( pid ) = 1;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    double ys = inputs["yield_stress"];
    auto force_model = CabanaPD::createForceModel(
        model_type{}, mechanics_type{}, *particles, delta, K, G0, sigma_y );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd =
        CabanaPD::createSolver<memory_space>( inputs, particles, force_model );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Reset forces on both pins.
    std::vector<CabanaPD::RegionBoundary<CabanaPD::Cylinder>> cylinders = {
        cylinder1, cylinder2 };
    auto bc =
        createBoundaryCondition( CabanaPD::ForceValueBCTag{}, 0.0, exec_space{},
                                 *particles, cylinders, true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( prenotch );
    cabana_pd->run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    compactTensionTestExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
