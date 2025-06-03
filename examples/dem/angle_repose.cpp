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
    using model_type = CabanaPD::HertzianModel;
    model_type contact_model( radius, radius_extend, nu, E, e );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    double min_height = inputs["min_height"];
    double max_height = inputs["max_height"];
    double diameter = inputs["cylinder_diameter"];
    double cylinder_radius = 0.5 * diameter;
    double wall_thickness = inputs["wall_thickness"];
    double particle_radius = cylinder_radius - wall_thickness;
    auto create_container = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Only create particles for container.
        double rsq = x[0] * x[0] + x[1] * x[1];
        if ( ( x[2] > min_height && rsq > particle_radius * particle_radius &&
               rsq < cylinder_radius * cylinder_radius ) )
            return true;
        if ( ( rsq < particle_radius * particle_radius &&
               x[2] > high_corner[2] - wall_thickness ) )
            return true;
        return false;
    };
    CabanaPD::Particles particles(
        memory_space{}, model_type{}, CabanaPD::BaseOutput{}, low_corner,
        high_corner, num_cells, halo_width, Cabana::InitRandom{},
        create_container, exec_space{}, true );
    auto create = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Only create particles inside cylinder.
        double rsq = x[0] * x[0] + x[1] * x[1];
        if ( x[2] > min_height && x[2] < max_height &&
             rsq < particle_radius * particle_radius )
            return true;
        return false;
    };
    particles.createParticles( exec_space{}, Cabana::InitRandom{}, create,
                               particles.localOffset() );

    // Set density/volumes.
    auto rho = particles.sliceDensity();
    auto vol = particles.sliceVolume();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        vol( pid ) = vol0;
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
    };
    CabanaPD::BodyTerm body( body_func, solver.particles.size(), true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.run( body );
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
