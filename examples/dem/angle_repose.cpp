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

// Simulate powder settling for angle of repose calculation.
void angleOfReposeExample( const std::string filename )
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
    double vol0 = inputs["volume"];
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
    int halo_width = 1;

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::HertzianJKRModel;
    model_type contact_model( radius, radius_extend, nu, E, e, gamma );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    double min_height = inputs["min_height"];
    double max_height = inputs["max_height"];
    double diameter = inputs["cylinder_diameter"];
    int feed_freq = inputs["feed_frequency"];
    double cylinder_radius = 0.5 * diameter;
    auto create = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Only create particles inside cylinder.
        double rsq = x[0] * x[0] + x[1] * x[1];
        if ( x[2] > min_height && x[2] < max_height &&
             rsq < cylinder_radius * cylinder_radius )
            return true;
        return false;
    };
    CabanaPD::Particles particles(
        memory_space{}, model_type{}, CabanaPD::BaseOutput{}, low_corner,
        high_corner, num_cells, halo_width, Cabana::InitRandom{}, create,
        exec_space{}, false );

    // Set density/volumes.
    auto rho = particles.sliceDensity();
    auto vol = particles.sliceVolume();
    auto vel = particles.sliceVelocity();
    auto y_init = particles.sliceCurrentPosition();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        vol( pid ) = vol0;
        vel( pid, 2 ) =
            -Kokkos::sqrt( 2.0 * 9.8 * ( max_height - y_init( pid, 2 ) ) );
    };
    particles.updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, contact_model );
    solver.init();

    // ====================================================
    //                   Boundary condition
    // ====================================================
    auto f = solver.particles.sliceForce();
    auto v = solver.particles.sliceVelocity();
    auto y = solver.particles.sliceCurrentPosition();
    auto body_func = KOKKOS_LAMBDA( const int p, const double )
    {
        // Gravity force.
        f( p, 2 ) -= 9.8 * rho( p );

        // Interact with a horizontal wall.
        double rz = y( p, 2 );
        if ( rz - radius < 0.0 || rz > high_corner[2] - radius )
        {
            double vz = v( p, 2 );
            double vn = rz * vz;
            vn /= rz;

            f( p, 2 ) +=
                -contact_model.forceCoeff( rz + radius, vn, vol0, rho0 );
        }

        // Interact with a cylindrical tube
        double r = Kokkos::hypot( y( p, 0 ), y( p, 1 ) );
        if ( rz > min_height && cylinder_radius + radius - r < 0.0 )
        {
            // Get components of unit normal vector
            double nx = -y( p, 0 ) / r;
            double ny = -y( p, 1 ) / r;

            // Normal velocity
            double vn = v( p, 0 ) * nx + v( p, 1 ) * ny;

            double fn = -contact_model.forceCoeff( cylinder_radius - r + radius,
                                                   vn, vol0, rho0 );

            f( p, 0 ) += fn * nx;
            f( p, 1 ) += fn * ny;
        }

        // Interact with vertical boundary walls.
        double rx = y( p, 0 ) >= 0.0 ? high_corner[0] - y( p, 0 )
                                     : y( p, 0 ) - low_corner[0];
        double ry = y( p, 1 ) >= 0.0 ? high_corner[1] - y( p, 1 )
                                     : y( p, 1 ) - low_corner[1];

        if ( rx - radius < 0.0 )
        {
            double nx = y( p, 0 ) >= 0.0 ? -1.0 : 1.0;
            double vn = v( p, 0 ) * nx;
            f( p, 0 ) +=
                -contact_model.forceCoeff( rx + radius, vn, vol0, rho0 );
        }
        if ( ry - radius < 0.0 )
        {
            double ny = y( p, 1 ) >= 0.0 ? -1.0 : 1.0;
            double vn = v( p, 1 ) * ny;
            f( p, 1 ) +=
                -contact_model.forceCoeff( ry + radius, vn, vol0, rho0 );
        }
    };
    CabanaPD::BodyTerm body( body_func, solver.particles.size(), true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    for ( int step = 1; step <= solver.num_steps; ++step )
    {
        solver.runStep( step, body );
        if ( step % feed_freq == 0 )
        {
            particles.createParticles( exec_space{}, Cabana::InitRandom{},
                                       create, particles.localOffset() );
        }
    }
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    angleOfReposeExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
